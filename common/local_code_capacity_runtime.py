from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, Protocol, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from qldpc.codes import CSSCode, ClassicalCode
from qldpc.decoders import get_decoder_BP_OSD
from qldpc.objects import Pauli


ErrorType = Literal["X", "Z"]
BinaryVector = NDArray[np.uint8]
BinaryMatrix = NDArray[np.uint8]


def validate_probability(p: float) -> float:
    """Coerce a candidate error rate to float and check it is a probability.

    Args:
        p: Candidate physical error rate.

    Returns:
        ``p`` as a ``float`` in ``[0, 1]``.

    Raises:
        ValueError: If ``p`` is outside ``[0, 1]``.
    """
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Physical error rate must satisfy 0 <= p <= 1, got {p}")
    return p


def binary_vector(
    value: Sequence[int] | NDArray[np.integer],
    length: int,
    name: str,
) -> BinaryVector:
    """Validate, flatten and copy a binary vector of a required length.

    Args:
        value: Sequence or array of integers; flattened before checking.
        length: Required number of entries.
        name: Label used in error messages.

    Returns:
        A freshly allocated ``np.uint8`` array of shape ``(length,)``.

    Raises:
        ValueError: If the flat size is not ``length`` or any entry is not 0 or 1.
    """
    vector = np.asarray(value, dtype=int).reshape(-1)
    if vector.size != length:
        raise ValueError(f"{name} must have length {length}, got {vector.size}")
    if not np.isin(vector, [0, 1]).all():
        raise ValueError(f"{name} must be binary over GF(2)")
    return vector.astype(np.uint8, copy=True)


@dataclass(frozen=True)
class DecodingProblem:
    """Select one binary X- or Z-error channel from a binary CSSCode."""

    code: CSSCode
    error_type: ErrorType | str
    check_matrix: BinaryMatrix = field(init=False, repr=False)
    stabilizer_matrix: BinaryMatrix = field(init=False, repr=False)
    dual_logicals: BinaryMatrix = field(init=False, repr=False)
    check_rank: int = field(init=False)

    def __post_init__(self) -> None:
        """Normalize ``error_type`` and freeze the matrices for that channel.

        An X-error problem decodes against ``code.matrix_z`` with the Z logical
        operators as the dual observables; a Z-error problem decodes against
        ``code.matrix_x`` with the X logical operators. The selected check,
        stabilizer and logical matrices are copied into read-only ``uint8``
        arrays, and the check rank is cached for reachability tests.

        Raises:
            ValueError: If ``error_type`` is not "X"/"Z", the code is not binary,
                or the code is a subsystem code.
        """
        error_type = str(self.error_type).upper()
        if error_type not in {"X", "Z"}:
            raise ValueError(f"error_type must be 'X' or 'Z', got {self.error_type!r}")
        object.__setattr__(self, "error_type", cast(ErrorType, error_type))

        if self.code.field.order != 2:
            raise ValueError("This runner currently supports binary CSS codes only")
        if self.code.is_subsystem_code:
            raise ValueError("This runner expects a commuting CSS stabilizer code")

        if error_type == "X":
            check = self.code.matrix_z
            stabilizers = self.code.matrix_x
            dual_logicals = self.code.get_logical_ops(Pauli.Z)
            check_rank = self.code.code_z.rank
        else:
            check = self.code.matrix_x
            stabilizers = self.code.matrix_z
            dual_logicals = self.code.get_logical_ops(Pauli.X)
            check_rank = self.code.code_x.rank

        n = self.code.num_qudits
        check_array = np.asarray(check, dtype=np.uint8).reshape(-1, n).copy()
        stabilizer_array = np.asarray(stabilizers, dtype=np.uint8).reshape(-1, n).copy()
        logical_array = np.asarray(dual_logicals, dtype=np.uint8).reshape(-1, n).copy()
        check_array.setflags(write=False)
        stabilizer_array.setflags(write=False)
        logical_array.setflags(write=False)

        object.__setattr__(self, "check_matrix", check_array)
        object.__setattr__(self, "stabilizer_matrix", stabilizer_array)
        object.__setattr__(self, "dual_logicals", logical_array)
        object.__setattr__(self, "check_rank", int(check_rank))

    @property
    def n(self) -> int:
        """Number of physical qudits in the code."""
        return int(self.code.num_qudits)

    @property
    def k(self) -> int:
        """Number of logical qudits protected by the code."""
        return int(self.code.dimension)

    @property
    def num_checks(self) -> int:
        """Number of stabilizer checks used as syndrome bits."""
        return int(self.check_matrix.shape[0])

    def validate_error(self, error: Sequence[int] | NDArray[np.integer]) -> BinaryVector:
        """Coerce user input into a valid physical-error vector.

        Args:
            error: Candidate error of length ``n``.

        Returns:
            Binary ``np.uint8`` vector of length ``n``.

        Raises:
            ValueError: If the length or entries are invalid.
        """
        return binary_vector(error, self.n, "error")

    def validate_syndrome(
        self,
        syndrome: Sequence[int] | NDArray[np.integer],
    ) -> BinaryVector:
        """Coerce user input into a valid syndrome vector.

        Args:
            syndrome: Candidate syndrome of length ``num_checks``.

        Returns:
            Binary ``np.uint8`` vector of length ``num_checks``.

        Raises:
            ValueError: If the length or entries are invalid.
        """
        return binary_vector(syndrome, self.num_checks, "syndrome")

    def syndrome(self, error: Sequence[int] | NDArray[np.integer]) -> BinaryVector:
        """Compute the syndrome ``H @ e mod 2`` for a physical error.

        Args:
            error: Physical error of length ``n``.

        Returns:
            Binary ``np.uint8`` syndrome vector of length ``num_checks``.
        """
        error_vector = self.validate_error(error)
        return ((self.check_matrix @ error_vector) % 2).astype(np.uint8)

    def syndrome_is_reachable(
        self,
        syndrome: Sequence[int] | NDArray[np.integer],
    ) -> bool:
        """Test whether a syndrome can be produced by some physical error.

        The syndrome is reachable when augmenting the check matrix with it does
        not increase the rank over GF(2).

        Args:
            syndrome: Candidate syndrome of length ``num_checks``.

        Returns:
            ``True`` if the syndrome lies in the column space of the check matrix.
        """
        syndrome_vector = self.validate_syndrome(syndrome)
        augmented = np.column_stack([self.check_matrix, syndrome_vector])
        augmented_rank = ClassicalCode(augmented, field=2).rank
        return int(augmented_rank) == self.check_rank

    def logical_signature(
        self,
        residual: Sequence[int] | NDArray[np.integer],
    ) -> BinaryVector:
        """Report which logical observables a residual error flips.

        Args:
            residual: Residual ``true_error XOR correction`` of length ``n``.

        Returns:
            Binary ``np.uint8`` vector of length ``k``; a nonzero entry means the
            residual anticommutes with that dual logical operator.
        """
        residual_vector = self.validate_error(residual)
        return ((self.dual_logicals @ residual_vector) % 2).astype(np.uint8)


class NoiseModel(Protocol):
    """Interface for a classical binary noise channel on physical qudits."""

    name: str
    error_type: ErrorType

    def sample(
        self,
        n: int,
        p: float,
        shots: int,
        rng: np.random.Generator,
    ) -> BinaryMatrix:
        """Draw ``shots`` i.i.d. errors.

        Args:
            n: Number of physical qudits per error.
            p: Physical error rate.
            shots: Number of samples to draw.
            rng: Random source.

        Returns:
            Binary matrix of shape ``(shots, n)``.
        """
        ...

    def probability(self, error: BinaryVector, p: float) -> float:
        """Return the likelihood of one specific error at rate ``p``.

        Args:
            error: Error vector of length ``n``.
            p: Physical error rate.

        Returns:
            Probability of observing exactly ``error``.
        """
        ...


def sample_binary_iid(
    n: int,
    p: float,
    shots: int,
    rng: np.random.Generator,
) -> BinaryMatrix:
    """Sample i.i.d. Bernoulli bit flips at rate ``p``.

    Args:
        n: Number of bits per sample.
        p: Probability that any single bit is 1.
        shots: Number of samples to draw.
        rng: Random source.

    Returns:
        Binary matrix of shape ``(shots, n)`` with dtype ``np.uint8``.

    Raises:
        ValueError: If ``p`` is not a probability or ``shots`` is not positive.
    """
    validate_probability(p)
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}")
    return (rng.random((shots, n)) < p).astype(np.uint8)


def binary_iid_probability(error: BinaryVector, p: float) -> float:
    """Compute the Bernoulli likelihood of one binary error vector.

    Args:
        error: Error vector of length ``n``.
        p: Probability that any single bit is 1.

    Returns:
        ``p ** weight * (1 - p) ** (n - weight)``, where the weight is the
        Hamming weight of ``error``.
    """
    p = validate_probability(p)
    weight = int(np.count_nonzero(error))
    return float((p**weight) * ((1.0 - p) ** (error.size - weight)))


@dataclass(frozen=True)
class BitFlipNoise:
    """IID X-error channel: each qubit flips independently with rate ``p``."""

    name: ClassVar[str] = "bit_flip"
    error_type: ClassVar[ErrorType] = "X"

    def sample(
        self,
        n: int,
        p: float,
        shots: int,
        rng: np.random.Generator,
    ) -> BinaryMatrix:
        """Draw ``shots`` independent X-error patterns.

        Args:
            n: Number of qudits.
            p: Per-qubit flip probability.
            shots: Number of samples.
            rng: Random source.

        Returns:
            Binary matrix of shape ``(shots, n)``.
        """
        return sample_binary_iid(n, p, shots, rng)

    def probability(self, error: BinaryVector, p: float) -> float:
        """Return the probability of exactly ``error`` at rate ``p``."""
        return binary_iid_probability(error, p)


@dataclass(frozen=True)
class PhaseFlipNoise:
    """IID Z-error channel: each qubit dephases independently with rate ``p``."""

    name: ClassVar[str] = "phase_flip"
    error_type: ClassVar[ErrorType] = "Z"

    def sample(
        self,
        n: int,
        p: float,
        shots: int,
        rng: np.random.Generator,
    ) -> BinaryMatrix:
        """Draw ``shots`` independent Z-error patterns.

        Args:
            n: Number of qudits.
            p: Per-qubit dephasing probability.
            shots: Number of samples.
            rng: Random source.

        Returns:
            Binary matrix of shape ``(shots, n)``.
        """
        return sample_binary_iid(n, p, shots, rng)

    def probability(self, error: BinaryVector, p: float) -> float:
        """Return the probability of exactly ``error`` at rate ``p``."""
        return binary_iid_probability(error, p)


class SyndromeDecoder(Protocol):
    """Interface for anything that maps a syndrome to a proposed correction."""

    def decode(self, syndrome: NDArray[np.integer]) -> NDArray[np.integer]:
        """Decode one syndrome.

        Args:
            syndrome: Binary syndrome of length ``num_checks``.

        Returns:
            Proposed correction of length ``n``.
        """
        ...


class DecoderFactory(Protocol):
    """Interface for building a decoder tuned to a given error rate."""

    name: str

    def build(self, problem: DecodingProblem, p: float) -> SyndromeDecoder:
        """Build a decoder for ``problem`` at physical error rate ``p``.

        Args:
            problem: Code/error-type pair the decoder will be used on.
            p: Physical error rate to tune the decoder for.

        Returns:
            A ready-to-use decoder.
        """
        ...


@dataclass(frozen=True)
class BPOSDDecoderFactory:
    """Create one qldpc BP+OSD decoder for each physical error rate."""

    max_iter: int = 50
    bp_method: str = "minimum_sum"
    schedule: str = "parallel"
    osd_method: str = "OSD_CS"
    osd_order: int = 2
    name: ClassVar[str] = "bp_osd"

    def __post_init__(self) -> None:
        """Reject nonsensical BP+OSD hyperparameters.

        Raises:
            ValueError: If ``max_iter`` is not positive or ``OSD_0`` is paired
                with a nonzero ``osd_order``.
        """
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.osd_method == "OSD_0" and self.osd_order != 0:
            raise ValueError("OSD_0 requires osd_order=0")

    def build(self, problem: DecodingProblem, p: float) -> SyndromeDecoder:
        """Instantiate a qldpc BP+OSD decoder for the problem's check matrix.

        Args:
            problem: Supplies the check matrix the decoder acts on.
            p: Physical error rate passed to BP+OSD as its channel estimate.

        Returns:
            A qldpc BP+OSD decoder taking syndromes and returning corrections.

        Raises:
            ValueError: If ``p`` is not in ``(0, 1)``, which BP+OSD requires.
        """
        p = validate_probability(p)
        if p in {0.0, 1.0}:
            raise ValueError("BP+OSD requires 0 < p < 1")
        return get_decoder_BP_OSD(
            problem.check_matrix,
            error_rate=p,
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            schedule=self.schedule,
            input_vector_type="syndrome",
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )


@dataclass(frozen=True)
class DecodeResult:
    """Outcome of a single decoder call on one syndrome.

    Attributes:
        p: Physical error rate the decoder was built for.
        syndrome: Input syndrome, length ``num_checks``.
        correction: Decoder-proposed correction, length ``n``.
        syndrome_valid: Whether the correction reproduces the input syndrome.
        converged: Decoder convergence flag if exposed, else ``None``.
        iterations: Decoder iteration count if exposed, else ``None``.
    """

    p: float
    syndrome: BinaryVector
    correction: BinaryVector
    syndrome_valid: bool
    converged: bool | None
    iterations: int | None


@dataclass(frozen=True)
class TrialResult:
    """Full bookkeeping for one sampled error through the decode pipeline.

    Attributes:
        p: Physical error rate used for sampling and decoding.
        true_error: The sampled physical error, length ``n``.
        true_error_probability: Likelihood of ``true_error`` under the noise model.
        syndrome: Syndrome produced by ``true_error``.
        correction: Decoder-proposed correction, length ``n``.
        residual: Bitwise XOR of ``true_error`` and ``correction``.
        logical_signature: Dual-logical measurement of ``residual``, length ``k``.
        syndrome_valid: Whether the correction reproduces ``syndrome``.
        logical_failure: ``syndrome_valid`` and ``residual`` flips a logical.
        total_failure: ``True`` if the syndrome is invalid or a logical failed.
        converged: Decoder convergence flag if exposed, else ``None``.
        iterations: Decoder iteration count if exposed, else ``None``.
    """

    p: float
    true_error: BinaryVector
    true_error_probability: float
    syndrome: BinaryVector
    correction: BinaryVector
    residual: BinaryVector
    logical_signature: BinaryVector
    syndrome_valid: bool
    logical_failure: bool
    total_failure: bool
    converged: bool | None
    iterations: int | None


@dataclass(frozen=True)
class SweepPoint:
    """Aggregated statistics for one ``(p, shots)`` Monte-Carlo point.

    Attributes:
        p: Physical error rate for this point.
        shots: Number of sampled errors.
        failures: Trials where the syndrome was invalid or a logical flipped.
        syndrome_invalid_failures: Trials whose correction missed the syndrome.
        logical_failures: Trials with a valid but logically wrong correction.
        logical_error_rate: ``failures / shots``.
        logical_error_rate_stderr: Binomial standard error of the rate.
        mean_error_weight: Mean Hamming weight of the sampled errors.
        mean_correction_weight: Mean Hamming weight of the corrections.
        mean_residual_weight: Mean Hamming weight of the residuals.
        decoder_converged_count: Trials reporting convergence, or ``None`` if the
            decoder exposes no convergence flag.
        decoder_converged_rate: ``decoder_converged_count / shots``, or ``None``.
        mean_decoder_iterations: Mean iteration count, or ``None`` if unavailable.
    """

    p: float
    shots: int
    failures: int
    syndrome_invalid_failures: int
    logical_failures: int
    logical_error_rate: float
    logical_error_rate_stderr: float
    mean_error_weight: float
    mean_correction_weight: float
    mean_residual_weight: float
    decoder_converged_count: int | None
    decoder_converged_rate: float | None
    mean_decoder_iterations: float | None


@dataclass(frozen=True)
class SweepResult:
    """A complete code-capacity sweep over several physical error rates.

    Attributes:
        code_name: Name of the simulated code.
        n: Number of physical qudits.
        k: Number of logical qudits.
        error_type: Error channel that was decoded ("X" or "Z").
        noise_model: Name reported by the noise model.
        decoder_name: Name reported by the decoder factory.
        seed: Master seed used to spawn per-point child seeds.
        points: One :class:`SweepPoint` per requested error rate, in order.
    """

    code_name: str
    n: int
    k: int
    error_type: ErrorType
    noise_model: str
    decoder_name: str
    seed: int
    points: tuple[SweepPoint, ...]

    def write_csv(self, path: str | Path) -> Path:
        """Write the sweep as a flat CSV, one row per point.

        Args:
            path: Destination file; parent directories are created if needed.

        Returns:
            The resolved output path.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "code",
            "n",
            "k",
            "error_type",
            "noise_model",
            "decoder",
            "seed",
            "p",
            "shots",
            "failures",
            "syndrome_invalid_failures",
            "logical_failures",
            "logical_error_rate",
            "logical_error_rate_stderr",
            "mean_error_weight",
            "mean_correction_weight",
            "mean_residual_weight",
            "decoder_converged_count",
            "decoder_converged_rate",
            "mean_decoder_iterations",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for point in self.points:
                writer.writerow(
                    {
                        "code": self.code_name,
                        "n": self.n,
                        "k": self.k,
                        "error_type": self.error_type,
                        "noise_model": self.noise_model,
                        "decoder": self.decoder_name,
                        "seed": self.seed,
                        "p": point.p,
                        "shots": point.shots,
                        "failures": point.failures,
                        "syndrome_invalid_failures": point.syndrome_invalid_failures,
                        "logical_failures": point.logical_failures,
                        "logical_error_rate": point.logical_error_rate,
                        "logical_error_rate_stderr": point.logical_error_rate_stderr,
                        "mean_error_weight": point.mean_error_weight,
                        "mean_correction_weight": point.mean_correction_weight,
                        "mean_residual_weight": point.mean_residual_weight,
                        "decoder_converged_count": point.decoder_converged_count,
                        "decoder_converged_rate": point.decoder_converged_rate,
                        "mean_decoder_iterations": point.mean_decoder_iterations,
                    }
                )
        return output_path


@dataclass
class CodeCapacityRunner:
    """Monte-Carlo code-capacity simulator: sample errors, decode, score failures.

    Attributes:
        problem: Code and error type under study.
        noise: Noise channel used to sample physical errors.
        decoder_factory: Builds one decoder per physical error rate.
        seed: Master seed; child seeds are spawned per sweep point.
        code_name: Label for reports; defaults to the code class name.
    """

    problem: DecodingProblem
    noise: NoiseModel
    decoder_factory: DecoderFactory
    seed: int = 1234
    code_name: str | None = None

    def __post_init__(self) -> None:
        """Check that the noise model matches the problem and fill in a name.

        Raises:
            ValueError: If the noise channel and the decoding problem disagree
                on the error type.
        """
        if self.noise.error_type != self.problem.error_type:
            raise ValueError(
                f"{self.noise.name} produces {self.noise.error_type} errors, but the "
                f"selected problem decodes {self.problem.error_type} errors"
            )
        if self.code_name is None:
            self.code_name = type(self.problem.code).__name__

    def _decode_with(
        self,
        decoder: SyndromeDecoder,
        syndrome: BinaryVector,
        p: float,
    ) -> DecodeResult:
        """Decode one syndrome and verify the returned correction.

        Args:
            decoder: Decoder to query, already built for rate ``p``.
            syndrome: Syndrome to decode.
            p: Physical error rate, recorded on the result.

        Returns:
            A :class:`DecodeResult` holding the validated correction, whether it
            reproduces the syndrome, and the decoder's convergence statistics.
        """
        correction = self.problem.validate_error(decoder.decode(syndrome))
        produced_syndrome = self.problem.syndrome(correction)
        syndrome_valid = bool(np.array_equal(produced_syndrome, syndrome))

        converged_value = getattr(decoder, "converge", None)
        iterations_value = getattr(decoder, "iter", None)
        converged = None if converged_value is None else bool(converged_value)
        iterations = None if iterations_value is None else int(iterations_value)

        return DecodeResult(
            p=p,
            syndrome=syndrome.copy(),
            correction=correction,
            syndrome_valid=syndrome_valid,
            converged=converged,
            iterations=iterations,
        )

    def _evaluate_error_with(
        self,
        decoder: SyndromeDecoder,
        true_error: BinaryVector,
        p: float,
    ) -> TrialResult:
        """Run one sampled error through the full syndrome-decode-score pipeline.

        Args:
            decoder: Decoder to query, already built for rate ``p``.
            true_error: The sampled physical error, length ``n``.
            p: Physical error rate, recorded on the result.

        Returns:
            A :class:`TrialResult` with the syndrome, correction, residual,
            logical signature and the resulting failure flags.
        """
        syndrome = self.problem.syndrome(true_error)
        decoded = self._decode_with(decoder, syndrome, p)
        residual = np.bitwise_xor(true_error, decoded.correction).astype(np.uint8)
        logical_signature = self.problem.logical_signature(residual)
        logical_failure = decoded.syndrome_valid and bool(np.any(logical_signature))
        total_failure = (not decoded.syndrome_valid) or logical_failure

        return TrialResult(
            p=p,
            true_error=true_error.copy(),
            true_error_probability=self.noise.probability(true_error, p),
            syndrome=syndrome,
            correction=decoded.correction,
            residual=residual,
            logical_signature=logical_signature,
            syndrome_valid=decoded.syndrome_valid,
            logical_failure=logical_failure,
            total_failure=total_failure,
            converged=decoded.converged,
            iterations=decoded.iterations,
        )

    def run_shots(
        self,
        p: float,
        shots: int,
        rng: np.random.Generator,
    ) -> SweepPoint:
        """Sample and decode ``shots`` errors at one physical error rate.

        Builds a fresh decoder for ``p``, samples that many errors, evaluates
        each trial, and aggregates failure counts, weights and decoder
        convergence statistics.

        Args:
            p: Physical error rate for this point.
            shots: Number of sampled errors to evaluate.
            rng: Random source for sampling.

        Returns:
            A :class:`SweepPoint` with the aggregated statistics.

        Raises:
            ValueError: If ``p`` is not a probability or ``shots`` is not positive.
        """
        p = validate_probability(p)
        if shots <= 0:
            raise ValueError(f"shots must be positive, got {shots}")

        decoder = self.decoder_factory.build(self.problem, p)
        errors = self.noise.sample(self.problem.n, p, shots, rng)
        trials = [
            self._evaluate_error_with(decoder, errors[index], p) for index in range(shots)
        ]

        failures = sum(trial.total_failure for trial in trials)
        syndrome_invalid = sum(not trial.syndrome_valid for trial in trials)
        logical_failures = sum(trial.logical_failure for trial in trials)
        logical_error_rate = failures / shots
        stderr = float(np.sqrt(logical_error_rate * (1.0 - logical_error_rate) / shots))

        converged_values = [
            trial.converged for trial in trials if trial.converged is not None
        ]
        iteration_values = [
            trial.iterations for trial in trials if trial.iterations is not None
        ]
        converged_count = sum(converged_values) if converged_values else None
        converged_rate = (
            converged_count / len(converged_values) if converged_count is not None else None
        )
        mean_iterations = float(np.mean(iteration_values)) if iteration_values else None

        return SweepPoint(
            p=p,
            shots=shots,
            failures=failures,
            syndrome_invalid_failures=syndrome_invalid,
            logical_failures=logical_failures,
            logical_error_rate=logical_error_rate,
            logical_error_rate_stderr=stderr,
            mean_error_weight=float(np.mean(errors.sum(axis=1))),
            mean_correction_weight=float(np.mean([trial.correction.sum() for trial in trials])),
            mean_residual_weight=float(np.mean([trial.residual.sum() for trial in trials])),
            decoder_converged_count=converged_count,
            decoder_converged_rate=converged_rate,
            mean_decoder_iterations=mean_iterations,
        )

    def sweep(self, p_values: Sequence[float], shots: int = 10_000) -> SweepResult:
        """Run a full code-capacity sweep across several error rates.

        Each rate gets its own decoder and its own child seed spawned from the
        master seed, so results are reproducible and independent of ordering.

        Args:
            p_values: Physical error rates to simulate.
            shots: Number of sampled errors per rate.

        Returns:
            A :class:`SweepResult` containing one point per rate, in order.

        Raises:
            ValueError: If ``p_values`` is empty or ``shots`` is not positive.
        """
        probabilities = [validate_probability(p) for p in p_values]
        if not probabilities:
            raise ValueError("p_values must not be empty")
        if shots <= 0:
            raise ValueError(f"shots must be positive, got {shots}")

        child_seeds = np.random.SeedSequence(self.seed).spawn(len(probabilities))
        points = tuple(
            self.run_shots(p, shots, np.random.default_rng(child_seed))
            for p, child_seed in zip(probabilities, child_seeds, strict=True)
        )
        assert self.code_name is not None
        return SweepResult(
            code_name=self.code_name,
            n=self.problem.n,
            k=self.problem.k,
            error_type=cast(ErrorType, self.problem.error_type),
            noise_model=self.noise.name,
            decoder_name=self.decoder_factory.name,
            seed=self.seed,
            points=points,
        )
