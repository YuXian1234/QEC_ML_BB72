from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np


CACHE_FORMAT_VERSION = 1


def parse_binary_vector(text: str, length: int, name: str) -> np.ndarray:
    """Parse a comma-separated binary string into a ``uint8`` vector.

    Args:
        text: Comma-separated 0/1 tokens, e.g. ``"1,0,1"``; blanks are ignored.
        length: Required number of entries (no. of rows).
        name: Label used in error messages.

    Returns:
        ``np.uint8`` array of shape ``(length,)``.

    Raises:
        ValueError: If the token count is not ``length`` or an entry is not
            binary.
    """
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    vector = np.asarray(values, dtype=np.uint8)
    if not np.isin(vector, [0, 1]).all():
        raise ValueError(f"{name} must be binary")
    return vector


def enumerate_masks(num_logicals: int) -> np.ndarray:
    """List every logical sector once, as a bit mask.
       Possible values of a's (4095)
    Args:
        num_logicals: Number of logical qubits ``k``.

    Returns:
        ``np.uint8`` array of shape ``(2**k, k)``; row ``m`` holds the bits of
        ``m`` little-endian, so every combination of logical flips appears once.
    """
    values = np.arange(1 << num_logicals, dtype=np.uint16)
    return ((values[:, None] >> np.arange(num_logicals, dtype=np.uint16)) & 1).astype(
        np.uint8
    )


def mask_to_bitstring(mask: np.ndarray) -> str:
    """Render a logical mask as a bit string for use in cache/report keys.

    Args:
        mask: Binary mask of ``k`` entries.

    Returns:
        One character per entry, e.g. ``"0101"``.
    """
    return "".join(str(int(bit)) for bit in mask.tolist())


def sector_index_from_bits(bits: np.ndarray) -> int:
    """Convert a logical mask back into its integer sector index.

    Uses the same little-endian convention as :func:`enumerate_masks`, so this
    is the inverse of indexing a row of that output.

    Args:
        bits: Binary mask of length ``k``.

    Returns:
        Integer in ``[0, 2**k)`` with bit ``i`` set when ``bits[i]`` is 1.
    """
    return int(sum(int(bit) << i for i, bit in enumerate(bits.tolist())))


def format_syndrome_labels(vector: np.ndarray) -> str:
    """Format the support of a syndrome as BB72 coordinate labels.

    Assumes the standard 6x6 torus layout, so detector ``i`` maps to lattice
    position ``(i // 6, i % 6)``.

    Args:
        vector: Syndrome vector of length ``num_checks``.

    Returns:
        String listing ``S_{row,col}`` for each nonzero entry.
    """
    labels = [f"S_{{{index // 6},{index % 6}}}" for index in np.flatnonzero(vector)]
    return str(labels)


def fwht_inplace(values: np.ndarray) -> None:
    """Apply the unnormalized Walsh-Hadamard transform in place.

    Each butterfly pass pairs entries ``step`` apart, replacing them with their
    sum and difference; the caller is responsible for the ``1 / N`` scaling.

    Args:
        values: 1-D float array whose length is a power of two. Modified in place.

    Returns:
        ``None``; ``values`` now holds the transformed spectrum.
    """
    length = values.shape[0]
    step = 1
    while step < length:
        jump = step << 1
        for start in range(0, length, jump):
            left = values[start : start + step].copy()
            right = values[start + step : start + jump].copy()
            values[start : start + step] = left + right
            values[start + step : start + jump] = left - right
        step = jump


def reconstruct_coset_probs(flip_probabilities: np.ndarray) -> np.ndarray:
    """Recover the full logical-sector distribution from single-flip marginals.

    Converts the per-logical flip probabilities into ``+/-1`` moments, applies
    the Walsh-Hadamard transform to invert the marginalisation, then clips tiny
    numerical negatives and renormalizes.

    Args:
        flip_probabilities: Probability that each logical is flipped, length
            ``k`` (a power of two).

    Returns:
        Non-negative ``float64`` array of length ``2**k`` summing to 1, indexed
        like the rows of :func:`enumerate_masks`.

    Raises:
        RuntimeError: If the transform yields a large negative probability or
            the result has no positive mass.
    """
    moments = 1.0 - 2.0 * flip_probabilities.astype(np.float64, copy=False)
    spectrum = moments.copy()
    fwht_inplace(spectrum)
    probabilities = spectrum / spectrum.size
    probabilities[np.abs(probabilities) < 1e-15] = 0.0
    min_probability = float(np.min(probabilities))
    if min_probability < -1e-10:
        raise RuntimeError(
            f"Walsh-Hadamard reconstruction produced a large negative value: {min_probability}"
        )
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise RuntimeError("Reconstructed probabilities have non-positive total mass")
    return probabilities / total


@contextlib.contextmanager
def suppress_output(enabled: bool):
    """Redirect stdout and stderr to ``os.devnull`` for the block's duration.

    Useful for hiding the chatter of CUDA-QX path optimization when it would
    bury the run's own progress output.

    Args:
        enabled: When ``False`` the block runs with output untouched.

    Yields:
        ``None``.
    """
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def probe_torch_cuda() -> dict[str, Any]:
    """Collect torch/CUDA availability details for run metadata.

    Never raises: an import failure or a failing CUDA query is recorded in the
    returned dictionary instead.

    Returns:
        Dict with keys ``torch_available``, ``torch_version``,
        ``torch_cuda_available``, ``torch_cuda_device_count`` and
        ``torch_cuda_device_name_0``, plus ``torch_import_error`` or
        ``torch_cuda_probe_error`` when something went wrong.
    """
    info: dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "torch_cuda_device_name_0": None,
    }
    try:
        import torch
    except Exception as exc:
        info["torch_import_error"] = repr(exc)
        return info

    info["torch_available"] = True
    info["torch_version"] = str(torch.__version__)
    try:
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_device_count"] = int(torch.cuda.device_count())
        if info["torch_cuda_available"] and info["torch_cuda_device_count"] > 0:
            info["torch_cuda_device_name_0"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        info["torch_cuda_probe_error"] = repr(exc)
    return info


def array_sha256(array: np.ndarray) -> str:
    """Hash an array's shape, dtype and contents for cache validation.

    Args:
        array: Any numpy array.

    Returns:
        Lowercase hex SHA-256 digest; two arrays collide only if they agree in
        shape, dtype and bytes.
    """
    normalized = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(normalized.shape).encode("ascii"))
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def build_cache_meta(
    problem: object,
    h_matrix: np.ndarray,
    logical_basis: np.ndarray,
    *,
    error_type: str,
    logical_basis_source: str,
    contractor_name: str,
) -> dict[str, Any]:
    """Assemble the metadata block that identifies a path cache.

    The metadata pins everything that would invalidate cached contraction paths:
    the code, the error channel, the dual logical basis (its source, its type and
    a hash of it), the contractor, the dtype, and hashes of the check matrix.

    Args:
        problem: Decoding problem supplying ``n``, ``num_checks`` and ``k``.
        h_matrix: Check matrix the decoder contracts against.
        logical_basis: Dual logical operators used as observables.
        error_type: Error channel being decoded ("X" or "Z").
        logical_basis_source: How the logical basis was obtained.
        contractor_name: Name of the CUDA-QX contractor in use.

    Returns:
        Dict of cache metadata keyed by field name.
    """
    logical_basis_type = "Z" if error_type == "X" else "X"
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "code": "[[72,12,6]]",
        "error_type": error_type,
        "logical_basis_source": logical_basis_source,
        "logical_basis_type": logical_basis_type,
        "contractor": contractor_name,
        "dtype": "float64",
        "contract_noise_model": True,
        "num_qubits": int(problem.n),
        "num_checks": int(problem.num_checks),
        "num_logicals": int(problem.k),
        "check_matrix_sha256": array_sha256(h_matrix),
        "logical_basis_sha256": array_sha256(logical_basis),
    }


def initialize_path_cache(meta: dict[str, Any]) -> dict[str, Any]:
    """Create an empty path cache carrying the given metadata.

    Args:
        meta: Metadata block from :func:`build_cache_meta`.

    Returns:
        Cache dict with an empty ``paths`` section, ready for new entries.
    """
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "meta": dict(meta),
        "paths": {},
    }


def cache_meta_matches(loaded_meta: dict[str, Any], expected_meta: dict[str, Any]) -> bool:
    """Check that a loaded cache was produced by an equivalent configuration.

    Args:
        loaded_meta: Metadata read from the cache file.
        expected_meta: Metadata for the current run.

    Returns:
        ``True`` if every expected field is present and equal in ``loaded_meta``.
    """
    for key, expected_value in expected_meta.items():
        if loaded_meta.get(key) != expected_value:
            return False
    return True


def load_path_cache(
    path: Path,
    expected_meta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a pickled path cache, falling back to an empty cache on any problem.

    Every failure mode (missing file, unreadable pickle, wrong structure, stale
    metadata) is non-fatal: an empty cache is returned so the run simply
    re-optimizes its paths.

    Args:
        path: Cache file to read.
        expected_meta: Metadata the cached entries must match.

    Returns:
        ``(cache, status)``. ``cache`` always has ``meta`` and ``paths``
        sections. ``status`` holds ``state`` (one of ``"loaded"``, ``"missing"``,
        ``"load-failed"``, ``"invalid-format"`` or ``"meta-mismatch"``),
        ``entries_loaded`` and a human-readable ``message``.
    """
    if not path.exists():
        return initialize_path_cache(expected_meta), {
            "state": "missing",
            "entries_loaded": 0,
            "message": f"cache file does not exist: {path}",
        }

    try:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
    except Exception as exc:
        return initialize_path_cache(expected_meta), {
            "state": "load-failed",
            "entries_loaded": 0,
            "message": f"failed to load cache {path}: {exc}",
        }

    if not isinstance(raw, dict):
        return initialize_path_cache(expected_meta), {
            "state": "invalid-format",
            "entries_loaded": 0,
            "message": f"cache {path} is not a dictionary",
        }

    loaded_meta = raw.get("meta")
    loaded_paths = raw.get("paths")
    if not isinstance(loaded_meta, dict) or not isinstance(loaded_paths, dict):
        return initialize_path_cache(expected_meta), {
            "state": "invalid-format",
            "entries_loaded": 0,
            "message": f"cache {path} is missing meta or paths sections",
        }

    if not cache_meta_matches(loaded_meta, expected_meta):
        return initialize_path_cache(expected_meta), {
            "state": "meta-mismatch",
            "entries_loaded": 0,
            "message": f"cache {path} metadata does not match the current run",
        }

    normalized_paths = {str(key): value for key, value in loaded_paths.items()}
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "meta": dict(expected_meta),
        "paths": normalized_paths,
    }, {
        "state": "loaded",
        "entries_loaded": len(normalized_paths),
        "message": f"loaded {len(normalized_paths)} cached paths from {path}",
    }


def atomic_pickle_dump(payload: dict[str, Any], path: Path) -> None:
    """Pickle a payload to ``path`` via a temp file and atomic replace.

    Writing to ``<path>.tmp`` first means an interrupted write cannot leave a
    truncated cache behind for the next run to trip over.

    Args:
        payload: Picklable object to persist.
        path: Destination file; parent directories are created if needed.

    Returns:
        ``None``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def optimize_decoder_path(decoder: object, optimize_arg: str | None) -> tuple[Any, float]:
    """Optimize the decoder's tensor-network contraction path, timing it.

    Args:
        decoder: CUDA-QX decoder exposing ``optimize_path(optimize=...)``.
        optimize_arg: Optimizer setting forwarded to CUDA-QX, or ``None`` for
            the decoder's own default.

    Returns:
        ``(path_info, elapsed_seconds)`` where ``path_info`` is whatever the
        decoder returned (typically a text summary of the chosen path).

    Raises:
        RuntimeError: If optimization fails, with a hint about CPU-only
            cuQuantum memory limits and the ``--skip-path-opt`` escape hatch.
    """
    path_start = time.perf_counter()
    try:
        path_info = decoder.optimize_path(optimize=optimize_arg)
    except RuntimeError as exc:
        raise RuntimeError(
            "optimize_path() failed. On CPU-only runs, CUDA-QX may require extra "
            "cuQuantum memory-limit configuration. Try a GPU node or rerun with "
            "--skip-path-opt if you only want a minimal smoke test."
        ) from exc
    return path_info, time.perf_counter() - path_start


def make_cache_entry(
    decoder: object,
    logical_obs: np.ndarray,
    path_info_text: str,
) -> dict[str, Any]:
    """Snapshot a decoder's optimized path into a cacheable dict.

    The entry records the contraction path and slicing, plus the weight and
    support of the logical observable it was optimized for, so a later run can
    check the entry still applies. The entry is round-tripped through pickle
    before being returned to fail early if the CUDA-QX path object cannot be
    serialized.

    Args:
        decoder: Decoder that has already been optimized.
        logical_obs: Logical observable array the path was optimized for.
        path_info_text: Textual description of the optimization result.

    Returns:
        Dict with ``path_single``, ``slicing_single``, ``logical_obs_weight``,
        ``logical_obs_support`` and ``path_info_text``.

    Raises:
        RuntimeError: If the decoder exposes no ``path_single``, or if the path
            object is not pickle-compatible.
    """
    path_single = getattr(decoder, "path_single", None)
    if path_single is None:
        raise RuntimeError("Decoder did not expose path_single after path optimization")
    entry = {
        "path_single": path_single,
        "slicing_single": getattr(decoder, "slicing_single", None),
        "logical_obs_weight": int(np.count_nonzero(logical_obs)),
        "logical_obs_support": np.flatnonzero(logical_obs.reshape(-1)).tolist(),
        "path_info_text": path_info_text,
    }
    try:
        pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise RuntimeError(
            "The optimized CUDA-QX path object could not be serialized into the cache file. "
            "This usually means the installed contractor path object is not pickle-compatible."
        ) from exc
    return entry


def restore_cached_path(decoder: object, entry: dict[str, Any]) -> None:
    """Install a cached contraction path onto a freshly built decoder.

    Args:
        decoder: Decoder to configure; modified in place, skipping the
            expensive path optimization.
        entry: Cache entry from :func:`make_cache_entry`.

    Returns:
        ``None``.
    """
    decoder.path_single = entry["path_single"]
    decoder.slicing_single = entry.get("slicing_single")


def raw_coset_masses_from_decoder(
    decoder: object,
    syndrome: np.ndarray,
    optimize_arg: str | None,
) -> tuple[float, float]:
    """Contract the tensor network to get the two logical coset masses.

    Bypasses the decoder's own probability normalization: it applies the
    syndrome, optimizes a path if none is cached, and contracts the network down
    to the single logical observable, returning the raw amplitudes for logical
    values 0 and 1.

    Args:
        decoder: CUDA-QX decoder whose tensor network is contracted.
        syndrome: Syndrome to condition on, length ``num_checks``.
        optimize_arg: Optimizer setting used if path optimization is needed.

    Returns:
        ``(z0, z1)`` raw masses for the two logical sectors; these are
        unnormalized and may be slightly negative.        Why?
    """
    syndrome_list = [float(x) for x in syndrome]
    decoder.flip_syndromes(syndrome_list)

    if decoder.path_single is None:
        decoder.optimize_path(optimize=optimize_arg)

    contraction = decoder.contractor_config.contractor(
        decoder.full_tn.get_equation(output_inds=(decoder.logical_obs_inds[0],)),
        decoder.full_tn.arrays,
        optimize=decoder.path_single,
        slicing=decoder.slicing_single,
        device_id=decoder.contractor_config.device_id,
    )
    z0, z1 = np.asarray(contraction, dtype=np.float64).reshape(2)
    return float(z0), float(z1)


def normalize_raw_coset_masses(
    z0: float,
    z1: float,
    *,
    relative_negative_tol: float = 1e-9,
    absolute_negative_tol: float = 1e-12,
) -> dict[str, Any]:
    """Turn raw tensor-network masses into a normalized two-outcome distribution.

    Negative masses from numerical error are clipped, but only while they stay
    within tolerance, so a genuinely broken contraction still surfaces as an
    error rather than silently becoming a small probability.

    Args:
        z0: Raw mass for logical value 0.
        z1: Raw mass for logical value 1.
        relative_negative_tol: Largest tolerated ``max_negative / scale``.
        absolute_negative_tol: Alternative pass condition for near-zero masses.

    Returns:
        Dict with the normalized probabilities ``p0``/``p1``, the original
        ``z0``/``z1``, the clipped ``z0_clipped``/``z1_clipped``, and the
        diagnostics ``clipped_mass_count``, ``max_negative_mass`` and
        ``relative_negative_mass``.

    Raises:
        RuntimeError: If the masses are non-finite, both zero, negative beyond
            tolerance, or have non-positive total after clipping.
    """
    raw = np.asarray([z0, z1], dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise RuntimeError(f"Raw TN masses are not finite: z0={z0}, z1={z1}")

    scale = float(np.max(np.abs(raw)))
    if scale == 0.0:
        raise RuntimeError("Raw TN masses are both zero")

    clipped = raw.copy()
    negative = np.maximum(-clipped, 0.0)
    max_negative = float(np.max(negative))
    relative_negative = max_negative / scale
    clipped_count = int(np.count_nonzero(negative > 0.0))

    if clipped_count:
        allowed = (
            max_negative <= absolute_negative_tol
            or relative_negative <= relative_negative_tol
        )
        if not allowed:
            raise RuntimeError(
                "Raw TN masses contain a negative value larger than the configured "
                f"tolerance: z0={z0}, z1={z1}, max_negative={max_negative}, "
                f"relative_negative={relative_negative}"
            )
        clipped = np.maximum(clipped, 0.0)

    total = float(np.sum(clipped))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError(
            f"Raw TN masses have non-positive total after clipping: z0={z0}, z1={z1}"
        )

    p0 = float(clipped[0] / total)
    p1 = float(clipped[1] / total)
    return {
        "p0": p0,
        "p1": p1,
        "z0": float(raw[0]),
        "z1": float(raw[1]),
        "z0_clipped": float(clipped[0]),
        "z1_clipped": float(clipped[1]),
        "clipped_mass_count": clipped_count,
        "max_negative_mass": max_negative,
        "relative_negative_mass": relative_negative,
    }


def query_flip_probability_with_fallback(
    decoder: object,
    syndrome: np.ndarray,
    optimize_arg: str | None,
    *,
    public_probability_tol: float = 1e-12,
    raw_relative_negative_tol: float = 1e-9,
    raw_absolute_negative_tol: float = 1e-12,
) -> dict[str, Any]:
    """Query the logical-flip probability, falling back to raw contraction.

    Tries the decoder's public ``decode`` first. If it raises, fails to
    converge, or returns a probability that is non-finite or outside ``[0, 1]``,
    the raw coset masses are contracted directly and normalized instead. Either
    way the returned dict carries timing and the rejected public result, so the
    caller can log why a fallback happened.

    Args:
        decoder: CUDA-QX decoder to query.
        syndrome: Syndrome to condition on, length ``num_checks``.
        optimize_arg: Optimizer setting used if a path must be optimized.
        public_probability_tol: Slack allowed when range-checking the public
            probability.
        raw_relative_negative_tol: Relative negative-mass tolerance passed to
            :func:`normalize_raw_coset_masses`.
        raw_absolute_negative_tol: Absolute negative-mass tolerance passed to
            :func:`normalize_raw_coset_masses`.

    Returns:
        Dict with ``p0``/``p1``, ``source`` (``"public"`` or ``"raw_fallback"``),
        the ``public_*`` diagnostic fields, the ``raw_*`` diagnostic fields
        (``None`` when unused), a ``fallback_reason``, and timings
        (``public_decode_seconds``, ``raw_fallback_seconds``,
        ``query_decode_seconds``).

    Raises:
        RuntimeError: If the raw fallback is used and
            :func:`normalize_raw_coset_masses` rejects the masses.
    """
    public_decode_seconds = 0.0
    raw_fallback_seconds = 0.0
    public_p1: float | None = None
    public_converged: bool | None = None
    public_exception: str | None = None
    fallback_reason: str | None = None

    public_start = time.perf_counter()
    try:
        public_result = decoder.decode([float(x) for x in syndrome])
        public_decode_seconds = time.perf_counter() - public_start
        public_p1 = float(np.asarray(public_result.result, dtype=np.float64).reshape(-1)[0])
        converged_value = getattr(public_result, "converged", None)
        public_converged = None if converged_value is None else bool(converged_value)
    except Exception as exc:
        public_decode_seconds = time.perf_counter() - public_start
        public_exception = repr(exc)
        fallback_reason = "public_exception"

    public_valid = False
    if public_exception is None:
        if public_converged is False:
            fallback_reason = "public_not_converged"
        elif public_p1 is None or not np.isfinite(public_p1):
            fallback_reason = "public_probability_not_finite"
        elif public_p1 < -public_probability_tol or public_p1 > 1.0 + public_probability_tol:
            fallback_reason = "public_probability_out_of_range"
        else:
            public_valid = True

    if public_valid and public_p1 is not None:
        p1 = float(np.clip(public_p1, 0.0, 1.0))
        return {
            "p1": p1,
            "p0": 1.0 - p1,
            "source": "public",
            "public_valid": True,
            "public_converged": public_converged,
            "public_p1": public_p1,
            "public_exception": public_exception,
            "fallback_reason": "",
            "raw_used": False,
            "raw_z0": None,
            "raw_z1": None,
            "raw_z0_clipped": None,
            "raw_z1_clipped": None,
            "raw_p0": None,
            "raw_p1": None,
            "raw_clipped_mass_count": 0,
            "raw_max_negative_mass": 0.0,
            "raw_relative_negative_mass": 0.0,
            "public_decode_seconds": public_decode_seconds,
            "raw_fallback_seconds": raw_fallback_seconds,
            "query_decode_seconds": public_decode_seconds,
        }

    raw_start = time.perf_counter()
    raw_z0, raw_z1 = raw_coset_masses_from_decoder(decoder, syndrome, optimize_arg)
    raw_info = normalize_raw_coset_masses(
        raw_z0,
        raw_z1,
        relative_negative_tol=raw_relative_negative_tol,
        absolute_negative_tol=raw_absolute_negative_tol,
    )
    raw_fallback_seconds = time.perf_counter() - raw_start

    return {
        "p1": raw_info["p1"],
        "p0": raw_info["p0"],
        "source": "raw_fallback",
        "public_valid": False,
        "public_converged": public_converged,
        "public_p1": public_p1,
        "public_exception": public_exception,
        "fallback_reason": fallback_reason or "public_invalid",
        "raw_used": True,
        "raw_z0": raw_info["z0"],
        "raw_z1": raw_info["z1"],
        "raw_z0_clipped": raw_info["z0_clipped"],
        "raw_z1_clipped": raw_info["z1_clipped"],
        "raw_p0": raw_info["p0"],
        "raw_p1": raw_info["p1"],
        "raw_clipped_mass_count": raw_info["clipped_mass_count"],
        "raw_max_negative_mass": raw_info["max_negative_mass"],
        "raw_relative_negative_mass": raw_info["relative_negative_mass"],
        "public_decode_seconds": public_decode_seconds,
        "raw_fallback_seconds": raw_fallback_seconds,
        "query_decode_seconds": public_decode_seconds + raw_fallback_seconds,
    }
