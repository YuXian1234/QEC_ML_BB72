from __future__ import annotations

from argparse import ArgumentParser
import inspect
from pathlib import Path

from qldpc.codes import BBCode
from sympy.abc import x, y

from common.local_code_capacity_runtime import BitFlipNoise, DecodingProblem, PhaseFlipNoise


DEFAULT_P_GRID = "0.005,0.01,0.02,0.03,0.04,0.05"


def build_bb72() -> BBCode:
    """Construct the standard [[72,12,6]] bivariate bicycle code.

    Uses the usual 6x6 torus layout with polynomials ``x^3 + y + y^2`` and
    ``y^3 + x + x^2``, then checks the resulting code really has 72 qudits and
    12 logical qudits.

    Returns:
        The verified :class:`BBCode`.

    Raises:
        RuntimeError: If the constructed code does not have parameters
            ``[[72, 12, 6]]``.
    """
    code = BBCode((6, 6), x**3 + y + y**2, y**3 + x + x**2)
    actual = (int(code.num_qudits), int(code.dimension))
    expected = (72, 12)
    if actual != expected:
        raise RuntimeError(f"BB72 construction mismatch: expected {expected}, got {actual}")
    return code


def build_problem(error_type: str) -> DecodingProblem:
    """Pair the BB72 code with one error channel.

    Args:
        error_type: ``"X"`` or ``"Z"`` (case-insensitive), selecting which
            check matrix and logical operators the problem decodes against.

    Returns:
        A :class:`DecodingProblem` for BB72.

    Raises:
        ValueError: If ``error_type`` is not ``"X"`` or ``"Z"``.
    """
    return DecodingProblem(build_bb72(), error_type)


def build_noise(error_type: str) -> BitFlipNoise | PhaseFlipNoise:
    """Select the noise model matching an error channel.

    Args:
        error_type: ``"X"`` or ``"Z"`` (case-insensitive).

    Returns:
        :class:`BitFlipNoise` for ``"X"``, :class:`PhaseFlipNoise` for ``"Z"``.

    Raises:
        ValueError: If ``error_type`` is not ``"X"`` or ``"Z"``.
    """
    normalized = str(error_type).upper()
    if normalized == "X":
        return BitFlipNoise()
    if normalized == "Z":
        return PhaseFlipNoise()
    raise ValueError(f"error_type must be 'X' or 'Z', got {error_type!r}")


def parse_p_grid(text: str) -> list[float]:
    """Parse a comma-separated list of physical error rates.

    Args:
        text: Rates as text, e.g. ``"0.01,0.02"``; blank tokens are ignored.

    Returns:
        The parsed rates as floats, in the order given.

    Raises:
        ValueError: If no usable value was found.
    """
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("p-grid must not be empty")
    return values


def default_results_dir(base_dir: Path | None = None) -> Path:
    """Return (creating it if needed) the ``results`` directory for a script.

    Args:
        base_dir: Directory to place ``results`` under. When ``None``, the
            directory of the calling script is used.

    Returns:
        Path to ``<base_dir>/results``.
    """
    if base_dir is None:
        caller = inspect.stack()[1].filename
        base_dir = Path(caller).resolve().parent
    path = base_dir / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_common_bb72_args(parser: ArgumentParser, *, default_seed: int) -> ArgumentParser:
    """Add the shared BB72 command-line options to a parser.

    Adds ``--error-type``, ``--p-grid``, ``--shots`` and ``--seed``.

    Args:
        parser: Argument parser to extend in place.
        default_seed: Value used for ``--seed`` when the flag is omitted, so
            each script keeps its own reproducible default.

    Returns:
        The same parser, for chaining.
    """
    parser.add_argument("--error-type", choices=["X", "Z"], default="X")
    parser.add_argument(
        "--p-grid",
        default=DEFAULT_P_GRID,
        help="Comma-separated physical error rates.",
    )
    parser.add_argument("--shots", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=default_seed)
    return parser
