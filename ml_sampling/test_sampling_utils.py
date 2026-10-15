"""Run with: python -m unittest discover -s . -p 'test_*.py' -v"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import sampling_utils as u
import sample_bb72_dataset as runner


def config():
    return {
        "sources": ["enumerate", "stress", "natural"], "enum_max_weight": 2,
        "stress_weights": [2, 3], "stress_shots_per_weight": 4,
        "natural_shots_per_p": 6, "p_grid": [0.001, 0.01],
        "seed": 123, "split_seed": 456, "max_relative_negative_mass": 1e-10,
        "max_relative_guard": 1e-8,
    }


def result(lam="000000000000"):
    return {
        "best_lambda": lam, "best_mass": 1.0, "root_mass": 2.0,
        "best_probability": 0.5, "wall_seconds": 0.1, "oracle_seconds": 0.09,
        "most_negative_raw_mass": -1e-15, "max_local_guard_abs": 2e-12,
        "clipped_negative_entries": 1, "fallbacks": [], "quality_flags": [],
    }


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.conn = u.open_database(self.root / "samples.sqlite")
        self.cfg = config()
        self.H = np.array([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.uint8)
        self.LZ = np.array([[1, 1, 0, 0]], dtype=np.uint8)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_default_candidate_count(self):
        cfg = config()
        cfg.update(p_grid=[.001, .002, .003, .005, .007, .01],
                   stress_weights=[3, 4, 5], stress_shots_per_weight=5000,
                   natural_shots_per_p=5000)
        self.assertEqual(u.candidate_count(cfg, 72), 47629)

    def test_deterministic_streams_and_exact_weights(self):
        first = list(u.candidate_errors(self.cfg, 4))
        second = list(u.candidate_errors(self.cfg, 4))
        self.assertEqual(len(first), 11 + 8 + 12)
        for a, b in zip(first, second):
            self.assertEqual(a[:2], b[:2])
            np.testing.assert_array_equal(a[2], b[2])
        self.assertEqual([int(e.sum()) for _, _, e in first[11:19]], [2]*4 + [3]*4)

    def test_natural_stream_independent_of_stress_count(self):
        other = dict(self.cfg, stress_shots_per_weight=20)
        a = [e for src, _, e in u.candidate_errors(self.cfg, 4) if src == "natural"]
        b = [e for src, _, e in u.candidate_errors(other, 4) if src == "natural"]
        np.testing.assert_array_equal(a, b)

    def test_dedup_preserves_conflicting_true_sectors(self):
        u.prepare_dataset(self.conn, self.cfg, self.H, self.LZ)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0], 31)
        self.assertLessEqual(len(u.selected_syndromes(self.conn, None)), 4)
        conflicts = self.conn.execute("SELECT syndrome_id FROM occurrences GROUP BY syndrome_id HAVING COUNT(DISTINCT true_sector_bits)>1").fetchall()
        self.assertTrue(conflicts)

    def test_generation_interruption_resume_matches_full(self):
        calls = [0]
        def stop():
            calls[0] += 1
            return calls[0] >= 7
        self.assertFalse(u.prepare_dataset(self.conn, self.cfg, self.H, self.LZ, stop))
        self.assertFalse(u.get_meta(self.conn, "generation_complete", False))
        u.prepare_dataset(self.conn, self.cfg, self.H, self.LZ)
        rows = [tuple(r) for r in self.conn.execute("SELECT * FROM occurrences ORDER BY ordinal")]
        full = u.open_database(self.root / "full.sqlite")
        try:
            u.prepare_dataset(full, self.cfg, self.H, self.LZ)
            self.assertEqual(rows, [tuple(r) for r in full.execute("SELECT * FROM occurrences ORDER BY ordinal")])
        finally:
            full.close()

    def test_completed_generation_is_not_repeated(self):
        u.prepare_dataset(self.conn, self.cfg, self.H, self.LZ)
        u.prepare_dataset(self.conn, self.cfg, self.H, self.LZ)
        self.assertEqual(u.get_meta(self.conn, "generation_cursor"), 31)

    def test_identity_rejects_changes(self):
        mats = {"H_Z": self.H, "L_Z": self.LZ}
        u.bind_config(self.conn, self.cfg, mats)
        u.bind_config(self.conn, self.cfg, mats)
        with self.assertRaises(ValueError):
            u.bind_config(self.conn, dict(self.cfg, seed=999), mats)
        changed = self.H.copy()
        changed[0, 0] ^= 1
        with self.assertRaises(ValueError):
            u.bind_config(self.conn, self.cfg, {"H_Z": changed, "L_Z": self.LZ})

    def test_running_task_recovered_but_done_preserved(self):
        u.prepare_dataset(self.conn, self.cfg, self.H, self.LZ)
        u.start_label(self.conn, 1, 0, .001)
        u.start_label(self.conn, 1, 1, .01)
        u.finish_label(self.conn, 1, 1, "done", result())
        self.assertEqual(u.recover_running(self.conn), 1)
        self.assertEqual([r[0] for r in self.conn.execute("SELECT status FROM labels ORDER BY p_index")], ["pending", "done"])
        u.start_label(self.conn, 1, 0, .001)
        self.assertEqual(self.conn.execute("SELECT attempts FROM labels WHERE p_index=0").fetchone()[0], 2)

    def test_quality_flags_and_nonfinite_rejected(self):
        good = result()
        self.assertEqual(u.assess_result(good, self.cfg)[0], "done")
        bad = result()
        bad["most_negative_raw_mass"] = -1e-6
        self.assertEqual(u.assess_result(bad, self.cfg)[0], "review")
        bad = result()
        bad["fallbacks"] = [{"reason": "OOM retry"}]
        self.assertIn("decoder_fallback", u.assess_result(bad, self.cfg)[1])
        for key in ("root_mass", "best_probability", "oracle_seconds", "most_negative_raw_mass", "max_local_guard_abs"):
            bad = result()
            bad[key] = float("nan")
            with self.assertRaises(ValueError):
                u.assess_result(bad, self.cfg)

    def test_bit_and_index_convention(self):
        self.assertEqual(u.sector_index([0, 1, 1, 1] + [0]*8), 14)
        with self.assertRaises(ValueError):
            u.bits([0, 2])

    def test_export_shapes_grouped_split_and_review_exclusion(self):
        H = np.zeros((36, 72), dtype=np.uint8)
        H[:, :36] = np.eye(36, dtype=np.uint8)
        L = np.zeros((12, 72), dtype=np.uint8)
        L[:, :12] = np.eye(12, dtype=np.uint8)
        cfg = dict(self.cfg, enum_max_weight=1, sources=["enumerate"], stress_weights=[3])
        u.bind_config(self.conn, cfg, {"H_Z": H, "L_Z": L})
        u.prepare_dataset(self.conn, cfg, H, L)
        for sid in (1, 2):
            for pi, p in enumerate(cfg["p_grid"]):
                u.start_label(self.conn, sid, pi, p)
                payload = result("011100000000")
                if sid == 2 and pi == 1:
                    payload["quality_flags"] = ["decoder_fallback"]
                u.finish_label(self.conn, sid, pi, "review" if payload["quality_flags"] else "done", payload)
        path = self.root / "ml_dataset.npz"
        self.assertEqual(u.export_ml(self.conn, path, cfg, 2), 3)
        with np.load(path, allow_pickle=False) as d:
            self.assertEqual(d["syndrome"].shape, (3, 36))
            self.assertEqual(d["map_sector_bits"].shape, (3, 12))
            np.testing.assert_array_equal(d["map_sector_index"], [14, 14, 14])
            self.assertEqual(d["split_id"][0], d["split_id"][1])
            self.assertEqual(d["natural_frequency"].shape, (3, 2))
            self.assertFalse(json.loads(str(d["metadata_json"]))["full_posterior_saved"])
        self.assertEqual(u.export_ml(self.conn, path, cfg, 2, include_review=True), 4)
        summary = u.summarize(self.conn, cfg, 2)
        self.assertTrue(summary["success"])
        self.assertFalse(summary["validation_passed"])

    def test_empty_export(self):
        p = self.root / "empty.npz"
        self.assertEqual(u.export_ml(self.conn, p, self.cfg), 0)
        with np.load(p, allow_pickle=False) as d:
            self.assertEqual(d["syndrome"].shape, (0, 36))

    def test_live_lock_refused_and_removed_normally(self):
        with u.RunLock(self.root):
            with self.assertRaises(RuntimeError):
                with u.RunLock(self.root, recover=True):
                    pass
        self.assertFalse((self.root / "sampling.lock").exists())

    def test_incomplete_lock_explicit_recovery(self):
        (self.root / "sampling.lock").write_text("", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            with u.RunLock(self.root):
                pass
        with u.RunLock(self.root, recover=True):
            self.assertTrue((self.root / "sampling.lock").exists())
        self.assertFalse((self.root / "sampling.lock").exists())

    def test_tiny_roundoff_clamped_but_raw_retained(self):
        cfg = {**self.cfg, "quality_policy_version": 2, "probability_boundary_atol": 1e-12}
        payload = result()
        payload.update(root_mass=261911726944288.0, best_mass=261911726944288.44,
                       best_probability=1.0000000000000018, most_negative_raw_mass=-.0078125,
                       max_local_guard_abs=261.911726944288)
        status, flags = u.assess_result(payload, cfg)
        self.assertEqual(status, "done")
        self.assertEqual(flags, [])
        self.assertEqual(payload["best_probability"], 1.0)
        self.assertEqual(payload["best_probability_raw"], 1.0000000000000018)
        self.assertTrue(payload["probability_boundary_corrected"])
        saved = u.canonical_json(payload)
        u.assess_result(payload, cfg)
        self.assertEqual(u.canonical_json(payload), saved)

    def test_large_overshoot_or_other_diagnostics_not_hidden(self):
        cfg = {**self.cfg, "quality_policy_version": 2, "probability_boundary_atol": 1e-12}
        for p, fallback in ((1.000001, []), (1.0000000000000018, [{"reason": "OOM"}])):
            payload = result()
            payload.update(best_probability=p, best_mass=2*p, fallbacks=fallback)
            status, flags = u.assess_result(payload, cfg)
            self.assertEqual(status, "review")
            self.assertIn("probability_outside_unit_interval", flags)
            self.assertEqual(payload["best_probability"], p)
            self.assertFalse(payload["probability_boundary_corrected"])


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.conn = u.open_database(self.root / "samples.sqlite")
        self.matrices = fake_matrices()
        self.old = {**config(), "sources": ["enumerate"], "enum_max_weight": 0,
                    "sampling_sha256": dict(u.LEGACY_SAMPLING_HASHES)}
        self.new = {**self.old, "sampling_sha256": {"sample_bb72_dataset.py": "new_runner", "sampling_utils.py": "new_utils"},
                    "quality_policy_version": 2, "probability_boundary_atol": 1e-12}
        sha = u.bind_config(self.conn, self.old, self.matrices)
        u.atomic_npz(self.root / "matrices.npz", **self.matrices)
        u.atomic_json(self.root / "run_manifest.json", {
            "config": self.old, "config_sha256": sha,
            "matrices_sha256": u.matrix_fingerprint(self.matrices)})
        u.prepare_dataset(self.conn, self.old, self.matrices["H_Z"], self.matrices["L_Z"])
        payload = result()
        payload.update(root_mass=261911726944288.0, best_mass=261911726944288.44,
                       best_probability=1.0000000000000018, most_negative_raw_mass=-.0078125,
                       max_local_guard_abs=261.911726944288)
        u.assess_result(payload, self.old)
        u.start_label(self.conn, 1, 0, .001)
        u.finish_label(self.conn, 1, 0, "review", payload)
        good = result()
        u.assess_result(good, self.old)
        u.start_label(self.conn, 1, 1, .01)
        u.finish_label(self.conn, 1, 1, "done", good)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_backup_audit_reclassify_and_idempotent_resume(self):
        with self.assertRaises(ValueError):
            u.bind_config(self.conn, self.new, self.matrices)
        upgraded = u.upgrade_roundoff_checkpoint(self.conn, self.new, self.matrices, self.root)
        self.assertEqual(upgraded["reclassified_labels"], 1)
        self.assertEqual(upgraded["boundary_corrected_labels"], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM label_quality_audit").fetchone()[0], 2)
        self.assertEqual(self.conn.execute("SELECT MAX(attempts) FROM labels").fetchone()[0], 1)
        backup = u.open_database(self.root / "samples.pre_roundoff_v1.sqlite")
        try:
            self.assertEqual(backup.execute("SELECT status FROM labels WHERE p_index=0").fetchone()[0], "review")
        finally:
            backup.close()
        # This also repairs a crash between the SQL commit and manifest replacement.
        u.sync_manifest(self.conn, self.root / "run_manifest.json", self.new, self.matrices)
        u.bind_config(self.conn, self.new, self.matrices)
        u.upgrade_roundoff_checkpoint(self.conn, self.new, self.matrices, self.root)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM label_quality_audit").fetchone()[0], 2)
        self.assertEqual(u.export_ml(self.conn, self.root / "ml.npz", self.new), 2)
        with np.load(self.root / "ml.npz", allow_pickle=False) as d:
            self.assertEqual(d["map_probability"][0], 1.0)
            self.assertGreater(d["map_probability_raw"][0], 1.0)
            self.assertTrue(d["probability_boundary_corrected"][0])

    def test_changed_prior_profile_or_unknown_code_refused(self):
        for change in ({"p_grid": [.002, .01]}, {"profile_sha256": "changed"},
                       {"probability_boundary_atol": 1e-10}):
            with self.assertRaises(ValueError):
                u.upgrade_roundoff_checkpoint(self.conn, {**self.new, **change}, self.matrices, self.root)
        with self.conn:
            identity = u.get_meta(self.conn, "identity")
            identity["config"]["sampling_sha256"] = {"unknown": "code"}
            u.set_meta(self.conn, "identity", identity)
        with self.assertRaises(ValueError):
            u.upgrade_roundoff_checkpoint(self.conn, self.new, self.matrices, self.root)

    def test_review_with_other_flags_stays_review(self):
        row = self.conn.execute("SELECT result_json FROM labels WHERE p_index=0").fetchone()
        payload = json.loads(row[0])
        payload["fallbacks"] = [{"reason": "OOM"}]
        u.assess_result(payload, self.old)
        u.finish_label(self.conn, 1, 0, "review", payload)
        upgrade = u.upgrade_roundoff_checkpoint(self.conn, self.new, self.matrices, self.root)
        self.assertEqual(upgrade["reclassified_labels"], 0)
        self.assertEqual(self.conn.execute("SELECT status FROM labels WHERE p_index=0").fetchone()[0], "review")

    def test_corrupt_manifest_refused_without_label_changes(self):
        manifest = json.loads((self.root / "run_manifest.json").read_text())
        manifest["config_sha256"] = "tampered"
        u.atomic_json(self.root / "run_manifest.json", manifest)
        with self.assertRaises(ValueError):
            u.upgrade_roundoff_checkpoint(self.conn, self.new, self.matrices, self.root)
        self.assertEqual(self.conn.execute("SELECT status FROM labels WHERE p_index=0").fetchone()[0], "review")


def fake_matrices():
    H = np.zeros((36, 72), dtype=np.uint8)
    H[:, :36] = np.eye(36, dtype=np.uint8)
    L = np.zeros((12, 72), dtype=np.uint8)
    L[:, :12] = np.eye(12, dtype=np.uint8)
    return {"H_Z": H, "H_X": H.copy(), "L_Z": L, "L_X": L.copy()}


class FakeRuntime:
    EASY_REFERENCE_P = .9999999955333834
    HARD_REFERENCE_P = .3333164874732356
    HARD_DEGENERATE_SECTORS = {"111011000000"}

    @staticmethod
    def make_case(H, LZ, positions):
        e = np.zeros(72, dtype=np.uint8)
        e[positions] = 1
        return {"syndrome": (H @ e) & 1}


class FakeTeacher:
    calls = []
    fail_next = False

    def __init__(self, production, matrices, p, profile, config):
        self.p = p
        self.runtime = FakeRuntime
        self.environment = {"gpu": "FAKE TEST GPU", "device_id": 0, "packages": {}}

    def decode(self, syndrome):
        self.calls.append((self.p, u.bits(syndrome)))
        if self.fail_next:
            type(self).fail_next = False
            raise ValueError("Injected failure")
        payload = result()
        positions = np.flatnonzero(syndrome).tolist()
        if positions == [28]:
            payload["best_probability"] = FakeRuntime.EASY_REFERENCE_P
        elif positions == [0, 27, 30]:
            payload["best_lambda"] = "111011000000"
            payload["best_probability"] = FakeRuntime.HARD_REFERENCE_P
        payload["best_mass"] = payload["root_mass"] * payload["best_probability"]
        return payload

    def close(self):
        pass


class RunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / "run"
        self.output.mkdir()
        FakeTeacher.calls = []
        self.base = ["--output-dir", str(self.output), "--sources", "enumerate",
                     "--enum-max-weight", "1", "--p-grid", "0.001,0.005"]

    def tearDown(self):
        self.tmp.cleanup()

    def execute(self, extra=()):
        args = runner.parser().parse_args(self.base + list(extra))
        with patch.object(runner, "build_bb72", side_effect=fake_matrices), patch.object(runner, "GpuTeacher", FakeTeacher), contextlib.redirect_stdout(io.StringIO()):
            with u.RunLock(self.output):
                return runner.run(args, self.output)

    def test_prefix_resume_and_expansion_no_duplicate_labels(self):
        self.assertEqual(self.execute(["--max-syndromes", "2"])["valid_labels"], 4)
        first = len(FakeTeacher.calls)
        self.assertEqual(self.execute(["--max-syndromes", "2"])["valid_labels"], 4)
        # A restart reruns only its two known preflight cases, not completed labels.
        self.assertEqual(len(FakeTeacher.calls) - first, 2)
        summary = self.execute(["--max-syndromes", "4"])
        self.assertEqual(summary["valid_labels"], 8)
        conn = u.open_database(self.output / "samples.sqlite")
        try:
            self.assertEqual(conn.execute("SELECT MAX(attempts) FROM labels").fetchone()[0], 1)
        finally:
            conn.close()

    def test_prepare_and_export_do_not_construct_gpu(self):
        with patch.object(runner, "build_bb72", side_effect=fake_matrices), patch.object(runner, "GpuTeacher", side_effect=AssertionError("GPU should not be constructed")):
            args = runner.parser().parse_args(self.base + ["--prepare-only", "--max-syndromes", "2"])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = runner.run(args, self.output)
            self.assertEqual(summary["expected_label_tasks"], 4)
            args = runner.parser().parse_args(["--output-dir", str(self.output), "--export-only"])
            summary = runner.run(args, self.output)
            self.assertEqual(summary["exported_ml_rows"], 0)

    def test_failed_task_recorded_and_retry_explicit(self):
        self.execute(["--max-syndromes", "1"])
        conn = u.open_database(self.output / "samples.sqlite")
        with conn:
            conn.execute("UPDATE labels SET status='failed',result_json=NULL,error='test failure' WHERE p_index=0")
        conn.close()
        self.assertEqual(self.execute(["--max-syndromes", "1"])["failed_tasks"], 1)
        summary = self.execute(["--max-syndromes", "1", "--retry-failed"])
        self.assertEqual(summary["failed_tasks"], 0)
        self.assertEqual(summary["valid_labels"], 2)

    def test_numeric_failure_continues_then_resumes_without_losing_good_labels(self):
        class OneFailure(FakeTeacher):
            injected = False

            def decode(self, syndrome):
                if self.p == .001 and np.flatnonzero(syndrome).tolist() == [0] and not type(self).injected:
                    type(self).injected = True
                    raise FloatingPointError("negative subtree mass exceeds abort tolerance")
                return super().decode(syndrome)

        args = runner.parser().parse_args(self.base + ["--max-syndromes", "3", "--continue-on-error"])
        with patch.object(runner, "build_bb72", side_effect=fake_matrices), patch.object(runner, "GpuTeacher", OneFailure), contextlib.redirect_stdout(io.StringIO()):
            summary = runner.run(args, self.output)
        self.assertEqual(summary["valid_labels"], 5)
        self.assertEqual(summary["failed_tasks"], 1)
        self.assertEqual(summary["pending_tasks"], 0)
        self.assertFalse(summary["success"])
        with np.load(self.output / "ml_dataset.npz", allow_pickle=False) as d:
            self.assertEqual(len(d["p_target"]), 5)
        self.assertEqual(self.execute(["--max-syndromes", "3"])["failed_tasks"], 1)
        summary = self.execute(["--max-syndromes", "3", "--retry-failed"])
        self.assertEqual(summary["valid_labels"], 6)
        conn = u.open_database(self.output / "samples.sqlite")
        try:
            self.assertEqual(conn.execute("SELECT attempts FROM labels WHERE syndrome_id=2 AND p_index=0").fetchone()[0], 2)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM labels WHERE attempts=1").fetchone()[0], 5)
        finally:
            conn.close()

    def test_config_mismatch_does_not_overwrite_exports(self):
        self.execute(["--max-syndromes", "1"])
        before = (self.output / "ml_dataset.npz").read_bytes()
        with self.assertRaises(ValueError):
            self.execute(["--seed", "999", "--max-syndromes", "1"])
        self.assertEqual(before, (self.output / "ml_dataset.npz").read_bytes())

    def test_cpu_upgrade_then_resume_without_recomputing_labels(self):
        args = runner.parser().parse_args(self.base + ["--prepare-only", "--max-syndromes", "1"])
        old = runner.make_config(args, runner.DEFAULT_PRODUCTION,
                                 runner.DEFAULT_PRODUCTION / "profiles" / "best_93.pkl")
        old.pop("quality_policy_version")
        old.pop("probability_boundary_atol")
        old["sampling_sha256"] = dict(u.LEGACY_SAMPLING_HASHES)
        with patch.object(runner, "build_bb72", side_effect=fake_matrices), patch.object(runner, "make_config", return_value=old), contextlib.redirect_stdout(io.StringIO()):
            runner.run(args, self.output)
        conn = u.open_database(self.output / "samples.sqlite")
        payload = result()
        payload.update(root_mass=261911726944288.0, best_mass=261911726944288.44,
                       best_probability=1.0000000000000018, most_negative_raw_mass=-.0078125,
                       max_local_guard_abs=261.911726944288)
        u.assess_result(payload, old)
        u.start_label(conn, 1, 0, .001)
        u.finish_label(conn, 1, 0, "review", payload)
        good = result()
        u.assess_result(good, old)
        u.start_label(conn, 1, 1, .005)
        u.finish_label(conn, 1, 1, "done", good)
        conn.close()
        args = runner.parser().parse_args(["--output-dir", str(self.output), "--export-only", "--upgrade-roundoff-checkpoint"])
        with patch.object(runner, "GpuTeacher", side_effect=AssertionError("No GPU needed for migration")), contextlib.redirect_stdout(io.StringIO()):
            summary = runner.run(args, self.output)
        self.assertEqual(summary["exported_ml_rows"], 2)
        self.assertEqual(summary["review_labels"], 0)
        self.assertEqual(summary["probability_boundary_corrected_labels"], 1)
        self.assertEqual(self.execute(["--prepare-only", "--max-syndromes", "1"])["valid_labels"], 2)


class GpuSafetyTests(unittest.TestCase):
    def test_no_cuda_is_an_error_not_cpu_fallback(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        cfg = config()
        profile = runner.DEFAULT_PRODUCTION / "profiles" / "best_93.pkl"
        cfg["profile_sha256"] = u.sha256_file(profile)
        with patch.dict(runner.sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "No CUDA GPU"):
                runner.GpuTeacher(runner.DEFAULT_PRODUCTION, fake_matrices(), .005,
                                  profile, cfg)

    def test_runtime_path_search_is_rejected(self):
        teacher = object.__new__(runner.GpuTeacher)
        teacher.decoder = SimpleNamespace(fast_cache_info=lambda: [{"path_source": "runtime_fallback_from_profile"}])
        with self.assertRaises(runner.ProfileError):
            teacher.require_compiled_paths()

    def test_only_expected_compiled_topology_sources_accepted(self):
        teacher = object.__new__(runner.GpuTeacher)
        teacher.decoder = SimpleNamespace(fast_cache_info=lambda: [{"path_source": "compiled_profile"}])
        teacher.require_compiled_paths()


if __name__ == "__main__":
    unittest.main()
