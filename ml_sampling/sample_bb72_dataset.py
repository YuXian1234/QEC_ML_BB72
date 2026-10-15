"""Generate BB72 syndromes and label them with the unchanged production GPU BnB."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import importlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import signal
import socket
import sys
import time

import numpy as np

from sampling_utils import (
    FORMAT_VERSION, SOURCE_ORDER, QUALITY_POLICY_VERSION, PROBABILITY_BOUNDARY_ATOL,
    RunLock, assess_result, atomic_json, atomic_npz,
    bind_config, bits, candidate_count, export_ml, finish_label, fingerprint,
    get_meta, matrix_fingerprint, open_database, prepare_dataset, recover_running,
    selected_syndromes, set_meta, sha256_file, start_label, summarize, utc_now,
    sync_manifest, upgrade_roundoff_checkpoint,
)


HERE = Path(__file__).resolve().parent
DEFAULT_PRODUCTION = HERE.parent / "v0.5_p0"
STOP = False


class ProfileError(RuntimeError):
    pass


def request_stop(signum, frame):
    global STOP
    STOP = True
    print(f"signal={signum}: stop requested; checkpoint after the current task", flush=True)


def versions() -> dict:
    names = ("numpy", "qldpc", "sympy", "cudaq-qec", "torch", "cupy-cuda12x",
             "cupy-cuda13x", "cuquantum-python-cu12", "cuquantum-python-cu13", "quimb", "autoray")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def build_bb72() -> dict[str, np.ndarray]:
    if versions()["qldpc"] != "0.3.3":
        raise RuntimeError("This frozen production profile requires qldpc==0.3.3; other versions can change the logical basis.")
    try:
        from sympy.abc import x, y
        from qldpc import codes
        from qldpc.objects import Pauli
    except ImportError as exc:
        raise RuntimeError("CPU dataset generation needs numpy, sympy and qldpc (the production basis uses qldpc 0.3.3).") from exc
    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    arrays = {
        "H_Z": np.asarray(code.matrix_z, dtype=np.uint8),
        "H_X": np.asarray(code.matrix_x, dtype=np.uint8),
        "L_Z": np.asarray(code.get_logical_ops(Pauli.Z), dtype=np.uint8),
        "L_X": np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8),
    }
    H, LZ, LX = arrays["H_Z"], arrays["L_Z"], arrays["L_X"]
    if H.shape != (36, 72) or LZ.shape != (12, 72):
        raise RuntimeError("Unexpected BB72 matrix dimensions")
    if not np.array_equal((LX @ LZ.T) & 1, np.eye(12, dtype=np.uint8)):
        raise RuntimeError("L_X L_Z^T != I")
    if np.any((arrays["H_X"] @ H.T) & 1) or np.any((arrays["H_X"] @ LZ.T) & 1):
        raise RuntimeError("Invalid CSS/logical commutation relations")
    return arrays


def profile_metadata(path: Path, memory_limit: str) -> dict:
    # Only load trusted, locally produced profiles; pickle is executable input.
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    meta = payload.get("metadata", {})
    paths = payload.get("paths", {})
    if tuple(range(9)) not in paths or (9, 10, 11) not in paths:
        raise ProfileError("Profile must contain both the b9 root and b3 tail topology")
    if meta.get("block_size", 9) != 9 or meta.get("logical_order", list(range(12))) != list(range(12)):
        raise ProfileError("Profile is not b9 identity")
    if str(meta.get("memory_limit", memory_limit)) != memory_limit:
        raise ProfileError("Use the profile's memory_limit, or a separately validated profile")
    return meta


class GpuTeacher:
    def __init__(self, production: Path, matrices: dict, p: float, profile: Path, config: dict):
        if sha256_file(profile) != config["profile_sha256"]:
            raise ProfileError("Profile changed during the experiment")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("GPU labeling requires the cudaqx-qec environment with CUDA-enabled torch.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU available. Use --prepare-only for CPU sampling; no CPU labeling fallback is allowed.")
        sys.path.insert(0, str(production))
        try:
            runtime = importlib.import_module("run_bb72_easy_hard")
        except ImportError as exc:
            raise RuntimeError(f"Cannot import the production GPU decoder ({exc}); activate the Linux cudaqx-qec environment.") from exc
        if Path(runtime.__file__).resolve() != (production / "run_bb72_easy_hard.py").resolve():
            raise RuntimeError("A different production runner is already imported in this process")
        self.runtime = runtime
        self.config = config
        self.decoder = None
        try:
            self.decoder, _ = runtime.make_decoder(
                matrices["H_Z"], matrices["L_Z"], p=p, profile=profile,
                memory_limit=config["memory_limit"],
            )
            cfg = self.decoder.contractor_config
            if cfg.contractor_name != "cutensornet" or not str(cfg.device).startswith("cuda"):
                raise RuntimeError(f"GPU cuTensorNet required, got {cfg}")
            self.environment = {
                "gpu": torch.cuda.get_device_name(int(cfg.device_id)),
                "device_id": int(cfg.device_id), "torch_cuda": torch.version.cuda,
                "contractor": cfg.contractor_name, "python_version": sys.version,
                "packages": versions(),
            }
            # Preparing both topologies makes runtime path-search fallback visible before labeling.
            for opened, fixed in ((list(range(9)), {}), ([9, 10, 11], dict.fromkeys(range(9), 0))):
                self.decoder.prepare_fast_logical_mass(
                    syndrome=[0.0] * 36, fixed_logicals=fixed, open_logicals=opened,
                    autotune_iterations=0, memory_limit=config["memory_limit"],
                    use_constant_qualifiers=True, use_gpu_resident_operands=True,
                )
            self.require_compiled_paths()
            self.decode(np.zeros(36, dtype=np.uint8))
        except BaseException:
            self.close()
            raise

    def require_compiled_paths(self):
        rows = self.decoder.fast_cache_info()
        if not rows or any(r["path_source"] != "compiled_profile" for r in rows):
            raise ProfileError(f"Specified compiled profile not actually used: {rows}")

    def decode(self, syndrome: np.ndarray) -> dict:
        fallbacks = []
        topology_calls = {}
        syndrome_list = syndrome.astype(float).tolist()

        def oracle(fixed, opened):
            self.runtime.sync_cuda()
            out = self.decoder.contract_logical_mass_fast(
                syndrome=syndrome_list, fixed_logicals=dict(fixed), open_logicals=list(opened),
                autotune_iterations=0, memory_limit=self.config["memory_limit"],
                release_workspace=True, use_constant_qualifiers=True,
                use_gpu_resident_operands=True, retain_constant_cache=True,
                retain_scratch_workspace=False, fallback_to_projector=False,
            )
            self.runtime.sync_cuda()
            self.require_compiled_paths()
            fallback = self.decoder.last_fast_fallback()
            if fallback is not None:
                fallbacks.append(fallback)
            key = "-".join(map(str, opened))
            topology_calls[key] = topology_calls.get(key, 0) + 1
            return out

        result = self.runtime.ExactBlockBnB(
            k=12, block_size=9, logical_order=list(range(12)), oracle=oracle,
            guard_rtol=self.config["guard_rtol"], print_tree=False,
        ).search()
        self.runtime.sync_cuda()
        return {
            **asdict(result.stats), "best_lambda": bits(result.best_lambda),
            "best_mass": float(result.best_mass), "root_mass": float(result.root_mass),
            "best_probability": float(result.best_probability), "fallbacks": fallbacks,
            "topology_calls": topology_calls, "path_source": "compiled_profile",
        }

    def close(self):
        if self.decoder is not None:
            self.decoder.clear_fast_network_cache()
            self.decoder = None
        gc.collect()
        self.runtime.cleanup_gpu()


def list_arg(text, cast):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--production-dir", type=Path, default=DEFAULT_PRODUCTION)
    ap.add_argument("--profile", type=Path, help="Trusted pickle; default: production/profiles/best_93.pkl")
    ap.add_argument("--output-dir", type=Path, default=HERE / "results" / "bb72_ml_small_p")
    ap.add_argument("--p-grid", default="0.001,0.002,0.003,0.005,0.007,0.01")
    ap.add_argument("--sources", default="enumerate,stress,natural")
    ap.add_argument("--enum-max-weight", type=int, default=2)
    ap.add_argument("--stress-weights", default="3,4,5")
    ap.add_argument("--stress-shots-per-weight", type=int, default=5000)
    ap.add_argument("--natural-shots-per-p", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--split-seed", type=int, default=20261015)
    ap.add_argument("--max-syndromes", type=int, help="Label only the first X unique syndromes; limit can grow on resume")
    ap.add_argument("--max-seconds", type=float, help="Soft stop after this many seconds, finishing the current task")
    ap.add_argument("--memory-limit", default="50%")
    ap.add_argument("--guard-rtol", type=float, default=1e-12)
    ap.add_argument("--max-relative-negative-mass", type=float, default=1e-10)
    ap.add_argument("--max-relative-guard", type=float, default=1e-8)
    ap.add_argument("--probability-boundary-atol", type=float, default=PROBABILITY_BOUNDARY_ATOL)
    ap.add_argument("--upgrade-roundoff-checkpoint", action="store_true",
                    help="Back up and upgrade the known original checkpoint without repeating TN labels")
    ap.add_argument("--prepare-only", action="store_true", help="Generate/deduplicate on CPU without importing CUDA-QX")
    ap.add_argument("--export-only", action="store_true", help="Export an existing checkpoint; only numpy is needed")
    ap.add_argument("--include-review", action="store_true", help="Explicitly include review labels in the NPZ export")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--retry-review", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true", help="Record failed tasks and rebuild the decoder before continuing")
    ap.add_argument("--recover-lock", action="store_true", help="Remove a stale lock ONLY after confirming the old job stopped")
    ap.add_argument("--summary-every", type=int, default=100)
    ap.add_argument("--export-every", type=int, default=0, help="0 exports only at exit; positive value exports periodically")
    return ap


def make_config(args, production, profile):
    ps = list_arg(args.p_grid, float)
    source_list = list_arg(args.sources, str)
    weights = list_arg(args.stress_weights, int)
    if not ps or len(set(ps)) != len(ps) or any(not np.isfinite(p) or not 0 < p < 0.5 for p in ps):
        raise ValueError("p-grid requires unique finite p values in (0,0.5)")
    if not source_list or set(source_list) - set(SOURCE_ORDER):
        raise ValueError(f"sources must be selected from {SOURCE_ORDER}")
    if not 0 <= args.enum_max_weight <= 3:
        raise ValueError("enum-max-weight must be 0..3 (avoid accidentally enumerating enormous sets)")
    if not weights or len(set(weights)) != len(weights) or any(not 0 <= w <= 72 for w in weights):
        raise ValueError("stress weights must be unique integers in 0..72")
    if args.stress_shots_per_weight < 0 or args.natural_shots_per_p < 0:
        raise ValueError("Shot counts must be nonnegative")
    if args.seed < 0 or args.split_seed < 0:
        raise ValueError("Seeds must be nonnegative")
    for value in (args.guard_rtol, args.max_relative_guard, args.max_relative_negative_mass,
                  args.probability_boundary_atol):
        if not np.isfinite(value) or value < 0:
            raise ValueError("Numerical tolerances must be finite and nonnegative")
    files = [production / name for name in (
        "exact_block_bnb.py", "multi_logical_tensor_network_decoder.py", "run_bb72_easy_hard.py")]
    config = {
        "code": "BB72 [[72,12,6]]", "channel": "iid X, perfect syndrome",
        "block_layout": [list(range(9)), [9, 10, 11]], "p_grid": ps,
        "sources": [s for s in SOURCE_ORDER if s in source_list],
        "enum_max_weight": args.enum_max_weight, "stress_weights": weights,
        "stress_shots_per_weight": args.stress_shots_per_weight,
        "natural_shots_per_p": args.natural_shots_per_p, "seed": args.seed,
        "split_seed": args.split_seed, "memory_limit": args.memory_limit,
        "guard_rtol": args.guard_rtol, "max_relative_negative_mass": args.max_relative_negative_mass,
        "max_relative_guard": args.max_relative_guard, "profile_sha256": sha256_file(profile),
        "production_sha256": {f.name: sha256_file(f) for f in files},
        "sampling_sha256": {name: sha256_file(HERE / name) for name in ("sample_bb72_dataset.py", "sampling_utils.py")},
        "qldpc_version": versions()["qldpc"],
        "sampling_numpy_version": np.__version__,
        "quality_policy_version": QUALITY_POLICY_VERSION,
        "probability_boundary_atol": args.probability_boundary_atol,
    }
    if candidate_count(config, 72) == 0:
        raise ValueError("Configuration generates no errors")
    return config


def upgrade_config(config, args):
    production = args.production_dir.resolve()
    profile = (args.profile or production / "profiles" / "best_93.pkl").resolve()
    if sha256_file(profile) != config["profile_sha256"]:
        raise ValueError("Cannot migrate labels with a different teacher profile")
    for name, sha in config["production_sha256"].items():
        if sha256_file(production / name) != sha:
            raise ValueError(f"Cannot migrate labels with different production code: {name}")
    return {**config,
            "sampling_sha256": {name: sha256_file(HERE / name) for name in ("sample_bb72_dataset.py", "sampling_utils.py")},
            "quality_policy_version": QUALITY_POLICY_VERSION,
            "probability_boundary_atol": args.probability_boundary_atol}


def write_outputs(conn, directory, config, args):
    summary = summarize(conn, config, args.max_syndromes)
    summary["exported_ml_rows"] = export_ml(
        conn, directory / "ml_dataset.npz", config, args.max_syndromes, args.include_review)
    summary["export_includes_review"] = args.include_review
    atomic_json(directory / "sampling_summary.json", summary)
    return summary


def run(args, directory):
    started = time.monotonic()
    def stopping():
        return STOP or (args.max_seconds is not None and time.monotonic() - started >= args.max_seconds)

    database = directory / "samples.sqlite"
    if args.export_only and not database.exists():
        raise FileNotFoundError("--export-only requires an existing experiment")
    conn = open_database(database)
    teacher = None
    config = None
    bound = False
    try:
        if args.export_only:
            identity = get_meta(conn, "identity")
            if identity is None:
                raise ValueError("No experiment identity in checkpoint")
            with np.load(directory / "matrices.npz", allow_pickle=False) as d:
                matrices = {name: d[name] for name in d.files}
            if matrix_fingerprint(matrices) != identity["matrices_sha256"]:
                raise ValueError("Saved matrices changed")
            config = identity["config"]
            if args.upgrade_roundoff_checkpoint:
                config = upgrade_config(config, args)
                upgrade = upgrade_roundoff_checkpoint(conn, config, matrices, directory)
                print("roundoff_upgrade=" + json.dumps({k: v for k, v in upgrade.items() if k != "old_identity"}), flush=True)
            sync_manifest(conn, directory / "run_manifest.json", config, matrices)
            bound = True
            return write_outputs(conn, directory, config, args)

        production = args.production_dir.resolve()
        profile = (args.profile or production / "profiles" / "best_93.pkl").resolve()
        profile_meta = profile_metadata(profile, args.memory_limit)
        matrices = build_bb72()
        config = make_config(args, production, profile)
        if args.upgrade_roundoff_checkpoint:
            upgrade = upgrade_roundoff_checkpoint(conn, config, matrices, directory)
            print("roundoff_upgrade=" + json.dumps({k: v for k, v in upgrade.items() if k != "old_identity"}), flush=True)
        sha = bind_config(conn, config, matrices)
        matrix_path = directory / "matrices.npz"
        if matrix_path.exists():
            with np.load(matrix_path, allow_pickle=False) as d:
                if matrix_fingerprint({key: d[key] for key in d.files}) != matrix_fingerprint(matrices):
                    raise ValueError("Saved matrix file differs from experiment")
        else:
            atomic_npz(matrix_path, **matrices)
        manifest_path = directory / "run_manifest.json"
        if not manifest_path.exists():
            atomic_json(manifest_path, {
                "format_version": FORMAT_VERSION, "created_at_utc": utc_now(),
                "config_sha256": sha, "config": config,
                "matrices_sha256": matrix_fingerprint(matrices),
                "profile_at_creation": str(profile), "profile_metadata": profile_meta,
                "production_at_creation": str(production),
                "sector_index_convention": "sum(lambda_i * 2**i); lambda_0 is first",
            })
        else:
            sync_manifest(conn, manifest_path, config, matrices)
        bound = True
        recovered = recover_running(conn)
        print(f"profile={profile}\nprofile_sha256={config['profile_sha256']}\nrecovered_tasks={recovered}", flush=True)
        if not prepare_dataset(conn, config, matrices["H_Z"], matrices["L_Z"], stopping):
            return write_outputs(conn, directory, config, args)
        selected = selected_syndromes(conn, args.max_syndromes)
        initial = summarize(conn, config, args.max_syndromes)
        print(json.dumps(initial, indent=2), flush=True)
        if args.prepare_only or stopping():
            return write_outputs(conn, directory, config, args)
        session = conn.execute("INSERT INTO sessions(started_at,environment_json) VALUES(?,?)",
                               (utc_now(), json.dumps({"host": socket.gethostname(), "python": sys.executable, "packages": versions()}))).lastrowid
        conn.commit()

        # Known reference probabilities apply ONLY at p=0.005, independently of the requested grid.
        teacher = GpuTeacher(production, matrices, 0.005, profile, config)
        previous_stack = get_meta(conn, "labeling_environment")
        stack = {k: v for k, v in teacher.environment.items() if k != "device_id"}
        if previous_stack is not None and previous_stack != stack:
            raise ValueError("GPU/software labeling environment changed; do not mix teacher labels in this experiment")
        preflight = []
        for name, positions in (("easy", [28]), ("hard", [0, 27, 30])):
            case = teacher.runtime.make_case(matrices["H_Z"], matrices["L_Z"], positions)
            result = teacher.decode(case["syndrome"])
            status, flags = assess_result(result, config)
            allowed = {"000000000000"} if name == "easy" else teacher.runtime.HARD_DEGENERATE_SECTORS
            reference = teacher.runtime.EASY_REFERENCE_P if name == "easy" else teacher.runtime.HARD_REFERENCE_P
            passed = status == "done" and result["best_lambda"] in allowed and abs(result["best_probability"] - reference) <= 1e-10
            preflight.append({"case": name, "passed": passed, "result": result})
        with conn:
            conn.execute("UPDATE sessions SET preflight_json=? WHERE id=?", (json.dumps(preflight), session))
        if not all(row["passed"] for row in preflight):
            raise RuntimeError("New-path easy/hard preflight failed; inspect sessions.preflight_json")
        with conn:
            set_meta(conn, "labeling_environment", stack)
        print(f"preflight=PASS device={teacher.environment['gpu']}", flush=True)
        teacher.close()
        teacher = None
        processed = 0
        for p_index, p in enumerate(config["p_grid"]):
            for row in selected:
                if stopping():
                    break
                existing = conn.execute("SELECT status FROM labels WHERE syndrome_id=? AND p_index=?", (row["id"], p_index)).fetchone()
                if existing is not None:
                    state = existing[0]
                    if state == "done" or (state == "review" and not args.retry_review) or (state == "failed" and not args.retry_failed):
                        continue
                if teacher is None:
                    teacher = GpuTeacher(production, matrices, p, profile, config)
                start_label(conn, row["id"], p_index, p)
                try:
                    syndrome = np.fromiter((int(b) for b in row["bits"]), dtype=np.uint8)
                    result = teacher.decode(syndrome)
                    result["session_id"] = session
                    status, flags = assess_result(result, config)
                    finish_label(conn, row["id"], p_index, status, result)
                except Exception as exc:
                    finish_label(conn, row["id"], p_index, "failed", error=f"{type(exc).__name__}: {exc}")
                    print(f"FAILED syndrome_id={row['id']} p={p}: {exc}", flush=True)
                    teacher.close()
                    teacher = None
                    if isinstance(exc, ProfileError) or not args.continue_on_error:
                        raise
                    status = "failed"
                processed += 1
                if processed <= 5 or processed % args.summary_every == 0:
                    print(f"labeled={processed} syndrome_id={row['id']} p={p} status={status}", flush=True)
                    atomic_json(directory / "sampling_summary.json", summarize(conn, config, args.max_syndromes))
                if args.export_every and processed % args.export_every == 0:
                    write_outputs(conn, directory, config, args)
            if teacher is not None:
                teacher.close()
                teacher = None
            if stopping():
                break
        return write_outputs(conn, directory, config, args)
    finally:
        try:
            if teacher is not None:
                teacher.close()
        finally:
            try:
                conn.rollback()
                if bound:
                    write_outputs(conn, directory, config, args)
            finally:
                conn.close()


def main() -> int:
    args = parser().parse_args()
    if args.max_syndromes is not None and args.max_syndromes < 1:
        raise SystemExit("--max-syndromes must be positive")
    if args.max_seconds is not None and (not np.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise SystemExit("--max-seconds must be positive and finite")
    if args.summary_every < 1 or args.export_every < 0:
        raise SystemExit("Invalid summary/export interval")
    if not np.isfinite(args.probability_boundary_atol) or not 0 <= args.probability_boundary_atol <= 1e-12:
        raise SystemExit("--probability-boundary-atol must be finite and in [0,1e-12]")
    if args.prepare_only and args.export_only:
        raise SystemExit("--prepare-only and --export-only are mutually exclusive")
    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with RunLock(directory, args.recover_lock):
        summary = run(args, directory)
    print(json.dumps(summary, indent=2), flush=True)
    if args.prepare_only or args.export_only:
        return 0
    return 0 if summary["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
