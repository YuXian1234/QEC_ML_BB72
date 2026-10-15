"""Read-only audit of the completed BB72 labeling dataset; never runs a decoder."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sampling_utils import matrix_fingerprint, fingerprint, sector_index, split_id


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for data in iter(lambda: f.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def rank2(a):
    a = a.copy().astype(np.uint8)
    r = 0
    for j in range(a.shape[1]):
        found = np.flatnonzero(a[r:, j])
        if not len(found):
            continue
        i = r + int(found[0])
        a[[i, r]] = a[[r, i]]
        for i in range(a.shape[0]):
            if i != r and a[i, j]:
                a[i] ^= a[r]
        r += 1
        if r == len(a):
            break
    return r


def distribution(xs):
    a = np.asarray(xs, dtype=float)
    if not len(a):
        return {"n": 0}
    return {"n": len(a), "min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "p95": float(np.quantile(a, .95)),
            "p99": float(np.quantile(a, .99)), "max": float(a.max())}


def audit(directory):
    files = [directory / n for n in ("samples.sqlite", "matrices.npz", "ml_dataset.npz",
                                     "run_manifest.json", "sampling_summary.json")]
    before = {p.name: digest(p) for p in files}
    manifest = json.loads((directory / "run_manifest.json").read_text())
    config = manifest["config"]
    summary = json.loads((directory / "sampling_summary.json").read_text())
    matrices = dict(np.load(directory / "matrices.npz", allow_pickle=False))
    conn = sqlite3.connect((directory / "samples.sqlite").as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()
    meta = {r[0]: json.loads(r[1]) for r in conn.execute("SELECT * FROM meta")}
    assert fingerprint(meta["identity"]) == manifest["config_sha256"] == summary["config_sha256"]
    assert meta["identity"]["config"] == config
    assert matrix_fingerprint(matrices) == manifest["matrices_sha256"]
    assert not np.any((matrices["H_X"] @ matrices["H_Z"].T) & 1)
    np.testing.assert_array_equal((matrices["L_X"] @ matrices["L_Z"].T) & 1, np.eye(12, dtype=np.uint8))
    assert not np.any((matrices["H_X"] @ matrices["L_Z"].T) & 1)
    syndromes = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM syndromes")}
    assert sorted(syndromes) == list(range(1, len(syndromes) + 1))
    assert len({r["bits"] for r in syndromes.values()}) == len(syndromes)
    for row in syndromes.values():
        assert len(row["bits"]) == 36 and row["bits"].count("1") == row["weight"]
        assert row["split_id"] == split_id(row["bits"], config["split_seed"])

    occurrences = [dict(r) for r in conn.execute("SELECT * FROM occurrences ORDER BY ordinal")]
    assert [r["ordinal"] for r in occurrences] == list(range(1, 47630))
    errors = np.array([[int(b) for b in r["error_bits"]] for r in occurrences], dtype=np.uint16)
    expected_s = np.array([[int(b) for b in syndromes[r["syndrome_id"]]["bits"]] for r in occurrences], dtype=np.uint8)
    expected_l = np.array([[int(b) for b in r["true_sector_bits"]] for r in occurrences], dtype=np.uint8)
    np.testing.assert_array_equal((errors @ matrices["H_Z"].T) & 1, expected_s)
    np.testing.assert_array_equal((errors @ matrices["L_Z"].T) & 1, expected_l)
    np.testing.assert_array_equal(errors.sum(axis=1), [r["error_weight"] for r in occurrences])
    source_counts = Counter(r["source"] for r in occurrences)
    assert dict(source_counts) == summary["source_record_counts"]
    by_sid = defaultdict(list)
    for row in occurrences:
        by_sid[row["syndrome_id"]].append(row)
    sources = {sid: {r["source"] for r in rows} for sid, rows in by_sid.items()}
    first_weight = {sid: rows[0]["error_weight"] for sid, rows in by_sid.items()}
    labels = [dict(r) for r in conn.execute("SELECT * FROM labels ORDER BY syndrome_id,p_index")]
    assert len(labels) == len(syndromes) * len(config["p_grid"]) == summary["expected_label_tasks"]
    assert len({(r["syndrome_id"], r["p_index"]) for r in labels}) == len(labels)
    label_map = {}
    by_p = defaultdict(Counter)
    flags = Counter()
    problems = []
    times = defaultdict(list)
    calls = defaultdict(Counter)
    by_weight = defaultdict(Counter)
    by_source = defaultdict(Counter)
    rel_negative = defaultdict(list)
    guards = defaultdict(list)
    classes = defaultdict(Counter)
    by_session = defaultdict(list)
    corrected = []
    for row in labels:
        sid, pi, p, status = row["syndrome_id"], row["p_index"], row["p_target"], row["status"]
        assert p == config["p_grid"][pi]
        assert status in ("done", "review", "failed")
        by_p[p][status] += 1
        by_weight[(p, first_weight[sid])][status] += 1
        for src in sources[sid]:
            by_source[(p, src)][status] += 1
        payload = json.loads(row["result_json"]) if row["result_json"] else None
        row["payload"] = payload
        label_map[sid, pi] = row
        if payload:
            assert (not payload["quality_flags"]) == (status == "done")
            assert len(payload["best_lambda"]) == 12 and not (set(payload["best_lambda"]) - {"0", "1"})
            assert np.isfinite(payload["best_probability"]) and payload["root_mass"] > 0
            assert payload["path_source"] == "compiled_profile"
            assert not payload.get("fallbacks")
            assert np.isclose(payload["best_probability_raw"], payload["best_mass"] / payload["root_mass"], rtol=1e-10, atol=1e-12)
            if status == "done":
                assert 0 <= payload["best_probability"] <= 1
                assert payload["relative_negative_mass"] <= config["max_relative_negative_mass"]
                assert payload["relative_guard"] <= config["max_relative_guard"]
                classes[p][payload["best_lambda"]] += 1
            flags.update(payload["quality_flags"])
            times[p].append(payload["wall_seconds"])
            calls[p][payload["oracle_calls"]] += 1
            rel_negative[p, status].append(payload["relative_negative_mass"])
            guards[p, status].append(payload["relative_guard"])
            by_session[payload["session_id"]].append(row)
            if payload["probability_boundary_corrected"]:
                corrected.append({"syndrome_id": sid, "p": p, "raw": payload["best_probability_raw"],
                                  "export_probability": payload["best_probability"]})
        else:
            assert status == "failed" and row["error"]
        if status != "done":
            example = {"syndrome_id": sid, "p_target": p, "status": status,
                       "syndrome_bits": syndromes[sid]["bits"], "diagnostic": payload,
                       "error": row["error"], "generating_occurrences": by_sid[sid]}
            problems.append(example)

    totals = Counter(r["status"] for r in labels)
    assert totals["done"] == summary["valid_labels"]
    assert totals["review"] == summary["review_labels"]
    assert totals["failed"] == summary["failed_tasks"]
    assert sum(len(t) for t in times.values()) == totals["done"] + totals["review"]
    assert np.isclose(sum(sum(t) for t in times.values()), summary["total_decode_seconds"], atol=1e-6)

    with np.load(directory / "ml_dataset.npz", allow_pickle=False) as archive:
        exported = {name: archive[name] for name in archive.files}
        metadata = json.loads(str(exported["metadata_json"]))
        assert metadata["config_sha256"] == manifest["config_sha256"]
        assert metadata["include_review"] is False
        valid = [r for r in labels if r["status"] == "done"]
        assert len(valid) == len(exported["p_target"]) == summary["exported_ml_rows"]
        assert len({(int(s), int(p)) for s, p in zip(exported["syndrome_id"], exported["p_index"])}) == len(valid)
        source_order = metadata["sources"]
        for i, row in enumerate(valid):
            sid, payload = row["syndrome_id"], row["payload"]
            assert exported["syndrome_id"][i] == sid and exported["p_index"][i] == row["p_index"]
            assert exported["p_target"][i] == row["p_target"]
            np.testing.assert_array_equal(exported["syndrome"][i], [int(b) for b in syndromes[sid]["bits"]])
            np.testing.assert_array_equal(exported["map_sector_bits"][i], [int(b) for b in payload["best_lambda"]])
            assert exported["map_sector_index"][i] == sector_index([int(b) for b in payload["best_lambda"]])
            assert exported["map_probability"][i] == payload["best_probability"]
            assert exported["map_probability_raw"][i] == payload["best_probability_raw"]
            assert exported["probability_boundary_corrected"][i] == payload["probability_boundary_corrected"]
            assert exported["probability_boundary_delta"][i] == payload["probability_boundary_delta"]
            assert json.loads(str(exported["quality_flags"][i])) == []
            assert exported["split_id"][i] == syndromes[sid]["split_id"]
            np.testing.assert_array_equal(exported["natural_frequency"][i], [
                sum(o["source"] == "natural" and o["generation_p"] == p for o in by_sid[sid]) for p in config["p_grid"]])
            np.testing.assert_array_equal(exported["source_frequency"][i], [
                sum(o["source"] == src for o in by_sid[sid]) for src in source_order])
        export_shapes = {k: list(exported[k].shape) for k in exported}
        y, split = exported["map_sector_index"], exported["split_id"]
        train_classes = set(y[split == 0].tolist())
        class_coverage = {
            "all_classes": len(set(y.tolist())), "train_classes": len(train_classes),
            "held_out": [{"split": i, "rows": int((split == i).sum()),
                          "classes_absent_from_training": len(set(y[split == i].tolist()) - train_classes),
                          "rows_with_classes_absent_from_training": int((~np.isin(y[split == i], list(train_classes))).sum())}
                         for i in (1, 2)],
        }

    natural_diagonal = defaultdict(Counter)
    natural_all_targets = defaultdict(Counter)
    for row in occurrences:
        if row["source"] != "natural":
            continue
        for pi, p in enumerate(config["p_grid"]):
            lab = label_map[row["syndrome_id"], pi]
            natural_all_targets[p][lab["status"]] += 1
            if row["generation_p"] == p:
                natural_diagonal[p][lab["status"]] += 1
                if lab["payload"]:
                    natural_diagonal[p]["true_sector_matches_teacher"] += int(row["true_sector_bits"] == lab["payload"]["best_lambda"])

    complete = []
    multi_sector = []
    n_good_by_s = Counter()
    for sid in syndromes:
        good = [label_map[sid, pi] for pi in range(6) if label_map[sid, pi]["status"] == "done"]
        n_good_by_s[len(good)] += 1
        if len(good) == 6:
            complete.append(sid)
        if len({r["payload"]["best_lambda"] for r in good}) > 1:
            multi_sector.append(sid)

    session_stats = []
    for session in conn.execute("SELECT * FROM sessions ORDER BY id"):
        rows = sorted(by_session[session["id"]], key=lambda r: r["updated_at"])
        intervals, unaccounted = [], []
        for previous, current in zip(rows, rows[1:]):
            dt = (datetime.fromisoformat(current["updated_at"]) - datetime.fromisoformat(previous["updated_at"])).total_seconds()
            intervals.append(dt)
            unaccounted.append(dt - current["payload"]["wall_seconds"])
        session_stats.append({"id": session["id"], "environment": json.loads(session["environment_json"]),
                              "successful_contractions": len(rows), "interval_seconds": distribution(intervals),
                              "interval_minus_current_decode_seconds": distribution(unaccounted),
                              "first_label_at": rows[0]["updated_at"] if rows else None,
                              "last_label_at": rows[-1]["updated_at"] if rows else None})

    source_sizes = {src: len({sid for sid, names in sources.items() if src in names}) for src in ("enumerate", "stress", "natural")}
    failure_details = []
    for problem in problems:
        if problem["status"] != "failed":
            continue
        match = re.search(r'negative subtree mass ([\deE.+-]+).*?root mass=([\deE.+-]+)', problem["error"])
        failure_details.append({"syndrome_id": problem["syndrome_id"], "p": problem["p_target"],
                                "syndrome_weight": problem["syndrome_bits"].count("1"),
                                "generators": problem["generating_occurrences"],
                                "negative_to_root": abs(float(match[1]))/float(match[2]) if match else None})
    result = {
        "audit_date": "2026-10-15", "source_directory": str(directory), "read_only": True,
        "source_sha256": before, "integrity_check": "ok", "all_assertions_passed": True,
        "ranks": {name: rank2(a) for name, a in matrices.items()},
        "counts": {"occurrences": len(occurrences), "unique_syndromes": len(syndromes),
                   "labels": len(labels), **dict(totals)},
        "by_p": [{"p": p, **dict(by_p[p]), "timing": distribution(times[p]),
                  "oracle_call_counts": dict(calls[p]), "distinct_map_sectors": len(classes[p]),
                  "most_frequent_map_sectors": classes[p].most_common(5)} for p in config["p_grid"]],
        "review_flag_counts": dict(flags), "boundary_corrections": corrected,
        "by_first_generating_weight": [{"p": p, "weight": w, **dict(v)} for (p, w), v in sorted(by_weight.items())],
        "by_source_nonexclusive": [{"p": p, "source": src, **dict(v)} for (p, src), v in sorted(by_source.items())],
        "source_unique_syndromes_nonexclusive": source_sizes,
        "natural_diagonal_occurrences": [{"p": p, **dict(v)} for p, v in sorted(natural_diagonal.items())],
        "natural_occurrences_at_all_target_priors": [{"p": p, **dict(v)} for p, v in sorted(natural_all_targets.items())],
        "negative_relative_by_status": [{"p": p, "status": status, **distribution(v)} for (p, status), v in sorted(rel_negative.items())],
        "guard_relative_by_status": [{"p": p, "status": status, **distribution(v)} for (p, status), v in sorted(guards.items())],
        "unique_syndromes_by_valid_prior_count": dict(n_good_by_s),
        "complete_six_prior_syndromes": len(complete),
        "syndromes_with_different_selected_sectors_across_valid_priors": len(multi_sector),
        "unique_syndromes_by_split": dict(Counter(r["split_id"] for r in syndromes.values())),
        "export_rows_by_split": dict(Counter(syndromes[r["syndrome_id"]]["split_id"] for r in labels if r["status"] == "done")),
        "export_shapes": export_shapes, "failed_details": failure_details, "sessions": session_stats,
        "map_class_coverage": class_coverage,
        "local_runtime_hash_matches": {
            **{name: digest(ROOT.parent / "v0.5_p0" / name) == sha
               for name, sha in config["production_sha256"].items()},
            **{name: digest(ROOT / name) == sha for name, sha in config["sampling_sha256"].items()},
            "best_93.pkl": digest(ROOT.parent / "v0.5_p0" / "profiles" / "best_93.pkl") == config["profile_sha256"],
        },
    }
    conn.close()
    after = {p.name: digest(p) for p in files}
    assert before == after, "Source files changed during audit"
    result["source_hashes_unchanged"] = True
    return result, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", type=Path, default=ROOT / "results" / "bb72_ml_small_p")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    result, problems = audit(args.directory.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output_dir / "problem_labels.jsonl").open("w", encoding="utf-8") as f:
        for row in problems:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({k: result[k] for k in ("counts", "by_p", "review_flag_counts",
          "by_first_generating_weight", "natural_diagonal_occurrences",
          "source_unique_syndromes_nonexclusive", "complete_six_prior_syndromes",
          "unique_syndromes_by_valid_prior_count", "export_rows_by_split",
          "syndromes_with_different_selected_sectors_across_valid_priors", "source_hashes_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
