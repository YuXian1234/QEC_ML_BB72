"""CPU-only sampling, transactional checkpoints, and non-pickle ML exports."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
from datetime import datetime, timezone
from typing import Iterator

import numpy as np


FORMAT_VERSION = 1
SOURCE_ORDER = ("enumerate", "stress", "natural")
QUALITY_POLICY_VERSION = 2
PROBABILITY_BOUNDARY_ATOL = 1e-12
LEGACY_SAMPLING_HASHES = {
    "sample_bb72_dataset.py": "9971b6925034ebb70ba9e775e3e0a1fd8b40214020ea398116bb2700b4357dad",
    "sampling_utils.py": "15064e80664ed4ed69b2703cc3232894935c3f58466fb0d3f20798b5057c1cb4",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def matrix_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        a = np.ascontiguousarray(array)
        h.update(canonical_json([name, a.shape, a.dtype.str]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    if os.name == "nt":
        # os.kill(pid, 0) is a console-control event on Windows, not a liveness probe.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True


class RunLock:
    """One writer per experiment, including on a shared cluster filesystem."""

    def __init__(self, directory: Path, recover: bool = False):
        self.path = directory / "sampling.lock"
        self.recover = recover
        self.token = f"{socket.gethostname()}:{os.getpid()}:{utc_now()}"

    def __enter__(self):
        if self.path.exists():
            try:
                owner = json.loads(self.path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError):
                if not self.recover:
                    raise RuntimeError("Incomplete lock file; confirm the old job stopped before --recover-lock")
                owner = {"owner_unknown": True}
            live = False
            if owner.get("host") == socket.gethostname():
                live = process_alive(int(owner["pid"]))
            if live or not self.recover:
                raise RuntimeError(
                    f"Experiment locked by {owner}. After confirming the old job "
                    "has stopped, use --recover-lock for a stale lock."
                )
            self.path.unlink()
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"host": socket.gethostname(), "pid": os.getpid(),
                       "started_at": utc_now(), "token": self.token}, fh)
        return self

    def __exit__(self, *args):
        if self.path.exists():
            owner = json.loads(self.path.read_text(encoding="utf-8"))
            if owner.get("token") == self.token:
                self.path.unlink()


def bits(array) -> str:
    a = np.asarray(array)
    if a.ndim != 1 or not np.all((a == 0) | (a == 1)):
        raise ValueError("Expected a one-dimensional binary vector")
    return "".join(str(int(x)) for x in a)


def sector_index(logical_bits) -> int:
    # The first vector element is lambda_0, the least-significant index bit.
    s = bits(logical_bits)
    return sum(int(b) << i for i, b in enumerate(s))


def split_id(syndrome: str, seed: int) -> int:
    bucket = int(hashlib.sha256(f"{seed}:{syndrome}".encode()).hexdigest()[:16], 16) % 10000
    return 0 if bucket < 8000 else (1 if bucket < 9000 else 2)


def candidate_errors(config: dict, n: int) -> Iterator[tuple[str, float | None, np.ndarray]]:
    """Fixed source order and independent streams make interrupted generation reproducible."""
    selected = set(config["sources"])
    if "enumerate" in selected:
        for w in range(config["enum_max_weight"] + 1):
            for support in itertools.combinations(range(n), w):
                e = np.zeros(n, dtype=np.uint8)
                e[list(support)] = 1
                yield "enumerate", None, e
    if "stress" in selected:
        for w in config["stress_weights"]:
            rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 1, w]))
            for _ in range(config["stress_shots_per_weight"]):
                e = np.zeros(n, dtype=np.uint8)
                e[rng.choice(n, size=w, replace=False)] = 1
                yield "stress", None, e
    if "natural" in selected:
        for i, p in enumerate(config["p_grid"]):
            rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 2, i]))
            for _ in range(config["natural_shots_per_p"]):
                yield "natural", p, (rng.random(n) < p).astype(np.uint8)


def candidate_count(config: dict, n: int) -> int:
    return (
        (sum(math.comb(n, w) for w in range(config["enum_max_weight"] + 1))
         if "enumerate" in config["sources"] else 0)
        + (len(config["stress_weights"]) * config["stress_shots_per_weight"]
           if "stress" in config["sources"] else 0)
        + (len(config["p_grid"]) * config["natural_shots_per_p"]
           if "natural" in config["sources"] else 0)
    )


def open_database(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    # DELETE journaling avoids WAL's shared-memory dependency on cluster filesystems.
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=FULL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS syndromes(
            id INTEGER PRIMARY KEY, bits TEXT UNIQUE NOT NULL,
            weight INTEGER NOT NULL, split_id INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS occurrences(
            ordinal INTEGER PRIMARY KEY, syndrome_id INTEGER NOT NULL REFERENCES syndromes(id),
            source TEXT NOT NULL, generation_p REAL, error_bits TEXT NOT NULL,
            true_sector_bits TEXT NOT NULL, error_weight INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS occurrence_syndrome ON occurrences(syndrome_id);
        CREATE TABLE IF NOT EXISTS labels(
            syndrome_id INTEGER NOT NULL REFERENCES syndromes(id), p_index INTEGER NOT NULL,
            p_target REAL NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            result_json TEXT, error TEXT, updated_at TEXT NOT NULL,
            PRIMARY KEY(syndrome_id,p_index));
        CREATE TABLE IF NOT EXISTS sessions(
            id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, environment_json TEXT NOT NULL,
            preflight_json TEXT);
        CREATE TABLE IF NOT EXISTS label_quality_audit(
            id INTEGER PRIMARY KEY, syndrome_id INTEGER NOT NULL, p_index INTEGER NOT NULL,
            old_status TEXT NOT NULL, old_result_json TEXT NOT NULL,
            new_status TEXT NOT NULL, new_result_json TEXT NOT NULL, changed_at TEXT NOT NULL);
    """)
    return conn


def get_meta(conn, key, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return default if row is None else json.loads(row[0])


def set_meta(conn, key, value) -> None:
    conn.execute("INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, canonical_json(value)))


def bind_config(conn, config: dict, matrices: dict[str, np.ndarray]) -> str:
    identity = {"format_version": FORMAT_VERSION, "config": config,
                "matrices_sha256": matrix_fingerprint(matrices)}
    sha = fingerprint(identity)
    old = get_meta(conn, "identity")
    if old is not None and old != identity:
        raise ValueError("Experiment configuration/matrices/code/profile changed. Use a new output directory, "
                         "or --upgrade-roundoff-checkpoint for the supported original quality-policy upgrade.")
    with conn:
        set_meta(conn, "identity", identity)
        set_meta(conn, "config_sha256", sha)
    return sha


def prepare_dataset(conn, config, H, LZ, stop_requested=lambda: False) -> bool:
    if get_meta(conn, "generation_complete", False):
        return True
    committed = get_meta(conn, "generation_cursor", 0)
    total = candidate_count(config, H.shape[1])
    for ordinal, (source, p, e) in enumerate(candidate_errors(config, H.shape[1]), 1):
        if ordinal <= committed:
            continue
        s = bits((H.astype(np.uint16) @ e.astype(np.uint16)) & 1)
        lam = bits((LZ.astype(np.uint16) @ e.astype(np.uint16)) & 1)
        row = conn.execute("SELECT id FROM syndromes WHERE bits=?", (s,)).fetchone()
        if row is None:
            sid = conn.execute(
                "INSERT INTO syndromes(bits,weight,split_id) VALUES(?,?,?)",
                (s, s.count("1"), split_id(s, config["split_seed"])),
            ).lastrowid
        else:
            sid = row[0]
        conn.execute("INSERT INTO occurrences VALUES(?,?,?,?,?,?,?)",
                     (ordinal, sid, source, p, bits(e), lam, int(e.sum())))
        set_meta(conn, "generation_cursor", ordinal)
        if ordinal % 250 == 0 or stop_requested():
            conn.commit()
        if stop_requested():
            return False
    with conn:
        set_meta(conn, "generation_cursor", total)
        set_meta(conn, "generation_complete", True)
    return True


def selected_syndromes(conn, limit: int | None):
    sql = "SELECT * FROM syndromes ORDER BY id"
    return conn.execute(sql if limit is None else sql + " LIMIT ?",
                        () if limit is None else (limit,)).fetchall()


def start_label(conn, sid: int, p_index: int, p: float) -> None:
    with conn:
        conn.execute("""INSERT INTO labels VALUES(?,?,?,'running',1,NULL,NULL,?)
            ON CONFLICT(syndrome_id,p_index) DO UPDATE SET
            status='running',attempts=attempts+1,result_json=NULL,error=NULL,updated_at=excluded.updated_at""",
            (sid, p_index, p, utc_now()))


def finish_label(conn, sid, p_index, status, result=None, error=None) -> None:
    if status not in ("done", "review", "failed"):
        raise ValueError(status)
    with conn:
        conn.execute("UPDATE labels SET status=?,result_json=?,error=?,updated_at=? WHERE syndrome_id=? AND p_index=?",
                     (status, None if result is None else canonical_json(result), error,
                      utc_now(), sid, p_index))


def recover_running(conn) -> int:
    with conn:
        cur = conn.execute("UPDATE labels SET status='pending',error='Interrupted; retry on resume' WHERE status='running'")
    return cur.rowcount


def assess_result(payload: dict, config: dict) -> tuple[str, list[str]]:
    flags = []
    raw_probability = float(payload.get("best_probability_raw", payload["best_probability"]))
    payload["best_probability"] = raw_probability
    if not math.isfinite(raw_probability):
        raise ValueError("Non-finite raw best probability")
    for name in ("best_mass", "root_mass", "best_probability", "wall_seconds", "oracle_seconds",
                 "most_negative_raw_mass", "max_local_guard_abs"):
        if not math.isfinite(float(payload[name])):
            raise ValueError(f"Non-finite {name}")
    if payload["root_mass"] <= 0 or payload["best_mass"] < 0:
        raise ValueError("Invalid TN masses")
    if payload["wall_seconds"] < 0 or payload["oracle_seconds"] < 0 or payload["max_local_guard_abs"] < 0:
        raise ValueError("Invalid timing/guard diagnostics")
    if not math.isclose(payload["best_probability"], payload["best_mass"] / payload["root_mass"],
                        rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Selected probability does not match best_mass/root_mass")
    lam = payload["best_lambda"]
    if len(lam) != 12 or set(lam) - {"0", "1"}:
        raise ValueError("Invalid logical sector")
    negative = abs(min(0.0, float(payload["most_negative_raw_mass"]))) / payload["root_mass"]
    guard = float(payload["max_local_guard_abs"]) / payload["root_mass"]
    if not math.isfinite(negative) or not math.isfinite(guard):
        raise ValueError("Non-finite numerical diagnostics")
    payload["relative_negative_mass"] = negative
    payload["relative_guard"] = guard
    if negative > config["max_relative_negative_mass"]:
        flags.append("relative_negative_mass_exceeded")
    if guard > config["max_relative_guard"]:
        flags.append("relative_guard_exceeded")
    if payload.get("fallbacks"):
        flags.append("decoder_fallback")
    corrected = False
    boundary_distance = max(0.0, -raw_probability, raw_probability - 1.0)
    tolerance = config.get("probability_boundary_atol", 0.0)
    if boundary_distance:
        if not flags and boundary_distance <= tolerance:
            payload["best_probability"] = min(1.0, max(0.0, raw_probability))
            corrected = True
        else:
            flags.append("probability_outside_unit_interval")
    payload["best_probability_raw"] = raw_probability
    payload["probability_boundary_corrected"] = corrected
    payload["probability_boundary_delta"] = payload["best_probability"] - raw_probability
    payload["quality_policy_version"] = config.get("quality_policy_version", 1)
    payload["quality_notes"] = ["probability_roundoff_clamped"] if corrected else []
    payload["quality_flags"] = flags
    return ("review" if flags else "done"), flags


def upgrade_roundoff_checkpoint(conn, config: dict, matrices: dict, directory: Path) -> dict:
    """Allow one audited, narrowly scoped upgrade; never whitelist arbitrary code changes."""
    old = get_meta(conn, "identity")
    new = {"format_version": FORMAT_VERSION, "config": config,
           "matrices_sha256": matrix_fingerprint(matrices)}
    if old is None:
        raise ValueError("No checkpoint to upgrade")
    if old == new:
        return get_meta(conn, "roundoff_upgrade", {})
    legacy = old["config"]
    with np.load(directory / "matrices.npz", allow_pickle=False) as saved_matrices:
        if matrix_fingerprint({key: saved_matrices[key] for key in saved_matrices.files}) != old["matrices_sha256"]:
            raise ValueError("Saved matrices changed; checkpoint cannot be upgraded")
    stripped = dict(config)
    for name in ("sampling_sha256", "quality_policy_version", "probability_boundary_atol"):
        stripped.pop(name, None)
    old_stripped = dict(legacy)
    old_stripped.pop("sampling_sha256", None)
    if (old["format_version"] != FORMAT_VERSION or old["matrices_sha256"] != new["matrices_sha256"]
            or legacy.get("sampling_sha256") != LEGACY_SAMPLING_HASHES
            or old_stripped != stripped
            or config.get("quality_policy_version") != QUALITY_POLICY_VERSION
            or config.get("probability_boundary_atol") != PROBABILITY_BOUNDARY_ATOL):
        raise ValueError("Unsupported checkpoint upgrade: only the known original sampling code and "
                         "unchanged dataset/teacher/profile may migrate to the 1e-12 roundoff policy.")
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_sha = fingerprint(old)
    if (manifest.get("config") != legacy or manifest.get("config_sha256") != old_sha
            or manifest.get("matrices_sha256") != old["matrices_sha256"]):
        raise ValueError("Original manifest does not match the checkpoint")
    backup_path = directory / "samples.pre_roundoff_v1.sqlite"
    if backup_path.exists():
        backup = sqlite3.connect(backup_path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            if get_meta(backup, "identity") != old:
                raise ValueError("Existing roundoff backup belongs to a different checkpoint")
        finally:
            backup.close()
    else:
        temporary_backup = backup_path.with_name(backup_path.name + ".tmp")
        backup = sqlite3.connect(temporary_backup)
        try:
            conn.backup(backup)
        finally:
            backup.close()
        os.replace(temporary_backup, backup_path)
    manifest_backup = directory / "run_manifest.pre_roundoff_v1.json"
    if manifest_backup.exists():
        if json.loads(manifest_backup.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing manifest backup differs from the original manifest")
    else:
        atomic_json(manifest_backup, manifest)
    rows = conn.execute("SELECT syndrome_id,p_index,status,result_json FROM labels WHERE status IN ('done','review')").fetchall()
    reclassified = corrected = 0
    migration = {"at_utc": utc_now(), "old_identity": old,
                 "old_config_sha256": old_sha, "new_config_sha256": fingerprint(new)}
    # Identity, labels and audit records commit together; a crash rolls back this upgrade.
    with conn:
        for row in rows:
            payload = json.loads(row["result_json"])
            status, _ = assess_result(payload, config)
            if row["status"] == "review" and status == "done":
                reclassified += 1
            corrected += int(payload["probability_boundary_corrected"])
            encoded = canonical_json(payload)
            conn.execute("INSERT INTO label_quality_audit VALUES(NULL,?,?,?,?,?,?,?)",
                         (row["syndrome_id"], row["p_index"], row["status"], row["result_json"],
                          status, encoded, utc_now()))
            conn.execute("UPDATE labels SET status=?,result_json=?,updated_at=? WHERE syndrome_id=? AND p_index=?",
                         (status, encoded, utc_now(), row["syndrome_id"], row["p_index"]))
        migration.update(reclassified_labels=reclassified, boundary_corrected_labels=corrected)
        set_meta(conn, "identity", new)
        set_meta(conn, "config_sha256", fingerprint(new))
        set_meta(conn, "roundoff_upgrade", migration)
    return migration


def sync_manifest(conn, path: Path, config: dict, matrices: dict) -> None:
    saved = json.loads(path.read_text(encoding="utf-8"))
    sha = get_meta(conn, "config_sha256")
    matrix_sha = matrix_fingerprint(matrices)
    if (saved.get("config_sha256") == sha and saved.get("config") == config
            and saved.get("matrices_sha256") == matrix_sha):
        return
    upgrade = get_meta(conn, "roundoff_upgrade", {})
    old = upgrade.get("old_identity", {})
    if (upgrade.get("new_config_sha256") != sha
            or saved.get("config_sha256") != upgrade.get("old_config_sha256")
            or saved.get("config") != old.get("config")
            or saved.get("matrices_sha256") != old.get("matrices_sha256")
            or matrix_sha != old.get("matrices_sha256")):
        raise ValueError("Saved manifest differs from the authoritative checkpoint identity")
    saved.update(config=config, config_sha256=sha, matrices_sha256=matrix_sha,
                 roundoff_upgrade={k: v for k, v in upgrade.items() if k != "old_identity"})
    atomic_json(path, saved)


def summarize(conn, config, limit: int | None) -> dict:
    selected = selected_syndromes(conn, limit)
    maximum = selected[-1]["id"] if selected else 0
    states = dict(conn.execute("SELECT status,COUNT(*) FROM labels WHERE syndrome_id<=? GROUP BY status",
                               (maximum,)).fetchall())
    expected = len(selected) * len(config["p_grid"])
    completed = states.get("done", 0) + states.get("review", 0)
    failed = states.get("failed", 0)
    pending = expected - completed - failed
    source_counts = dict(conn.execute("SELECT source,COUNT(*) FROM occurrences GROUP BY source").fetchall())
    times = [json.loads(r[0])["wall_seconds"] for r in conn.execute(
        "SELECT result_json FROM labels WHERE syndrome_id<=? AND status IN ('done','review')", (maximum,))]
    corrected = sum(bool(json.loads(r[0]).get("probability_boundary_corrected")) for r in conn.execute(
        "SELECT result_json FROM labels WHERE syndrome_id<=? AND status IN ('done','review')", (maximum,)))
    return {
        "generated_error_records": conn.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0],
        "generation_complete": get_meta(conn, "generation_complete", False),
        "unique_syndromes": conn.execute("SELECT COUNT(*) FROM syndromes").fetchone()[0],
        "selected_syndromes": len(selected), "expected_label_tasks": expected,
        "valid_labels": states.get("done", 0), "review_labels": states.get("review", 0),
        "probability_boundary_corrected_labels": corrected,
        "failed_tasks": failed, "pending_tasks": pending, "remaining_nonvalid_tasks": expected - states.get("done", 0),
        "success": get_meta(conn, "generation_complete", False) and pending == 0 and failed == 0,
        "validation_passed": expected > 0 and states.get("done", 0) == expected,
        "source_record_counts": source_counts, "p_grid": config["p_grid"],
        "median_decode_seconds": float(np.median(times)) if times else None,
        "total_decode_seconds": float(sum(times)), "updated_at_utc": utc_now(),
        "config_sha256": get_meta(conn, "config_sha256"),
    }


def export_ml(conn, path: Path, config: dict, limit: int | None = None, include_review=False) -> int:
    selected = selected_syndromes(conn, limit)
    maximum = selected[-1]["id"] if selected else 0
    statuses = "('done','review')" if include_review else "('done')"
    rows = conn.execute(f"""SELECT s.id,s.bits,s.split_id,l.p_index,l.p_target,l.status,l.result_json
        FROM labels l JOIN syndromes s ON s.id=l.syndrome_id
        WHERE s.id<=? AND l.status IN {statuses} ORDER BY s.id,l.p_index""", (maximum,)).fetchall()
    freq: dict[int, list[int]] = {}
    source_freq: dict[int, list[int]] = {}
    for row in conn.execute("SELECT syndrome_id,source,generation_p,COUNT(*) FROM occurrences GROUP BY syndrome_id,source,generation_p"):
        sid, source, p, count = row
        source_freq.setdefault(sid, [0] * 3)[SOURCE_ORDER.index(source)] += count
        if source == "natural":
            freq.setdefault(sid, [0] * len(config["p_grid"]))[config["p_grid"].index(p)] += count
    payloads = [json.loads(r["result_json"]) for r in rows]
    n = len(rows)
    arrays = {
        "syndrome": np.array([[int(b) for b in r["bits"]] for r in rows], dtype=np.uint8).reshape(n, 36),
        "p_target": np.array([r["p_target"] for r in rows], dtype=np.float64),
        "p_index": np.array([r["p_index"] for r in rows], dtype=np.uint8),
        "syndrome_id": np.array([r["id"] for r in rows], dtype=np.int64),
        "map_sector_bits": np.array([[int(b) for b in d["best_lambda"]] for d in payloads], dtype=np.uint8).reshape(n, 12),
        "map_sector_index": np.array([sector_index([int(b) for b in d["best_lambda"]]) for d in payloads], dtype=np.uint16),
        "map_probability": np.array([d["best_probability"] for d in payloads], dtype=np.float64),
        "map_probability_raw": np.array([d.get("best_probability_raw", d["best_probability"]) for d in payloads], dtype=np.float64),
        "probability_boundary_corrected": np.array([d.get("probability_boundary_corrected", False) for d in payloads], dtype=np.bool_),
        "probability_boundary_delta": np.array([d.get("probability_boundary_delta", 0.0) for d in payloads], dtype=np.float64),
        "split_id": np.array([r["split_id"] for r in rows], dtype=np.uint8),
        "quality_flags": np.array([canonical_json(d["quality_flags"]) for d in payloads], dtype=str),
        "natural_frequency": np.array([freq.get(r["id"], [0] * len(config["p_grid"])) for r in rows], dtype=np.int64).reshape(n, len(config["p_grid"])),
        "source_frequency": np.array([source_freq.get(r["id"], [0] * 3) for r in rows], dtype=np.int64).reshape(n, 3),
        "metadata_json": np.array(canonical_json({
            "format_version": FORMAT_VERSION, "config_sha256": get_meta(conn, "config_sha256"),
            "p_grid": config["p_grid"], "sources": list(SOURCE_ORDER),
            "sector_index": "sum(lambda_i * 2**i); lambda_0 is the first vector element",
            "splits": {"0": "train", "1": "validation", "2": "unseen_syndrome_test"},
            "include_review": include_review, "full_posterior_saved": False,
            "quality_policy_version": config.get("quality_policy_version", 1),
            "probability_boundary_atol": config.get("probability_boundary_atol", 0.0),
            "numerical_certificate": "float64 diagnostics, not an interval certificate",
        })),
    }
    atomic_npz(path, **arrays)
    return n
