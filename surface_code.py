"""Distance-three surface-code extraction, noise, decoding, and fault audits."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import combinations, product
from math import fsum, prod, sqrt
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from simulator import PauliFrame, PauliString, Statevector, _integer

# Code definition


def _pauli(axis: str, support: tuple[int, ...]) -> PauliString:
    return PauliString("".join(axis if q in support else "I" for q in range(9)))


class RotatedSurfaceCodeD3:
    """Nine data qubits, eight independent commuting +1 stabilizers.

    Row-major layout (q0 is still the least significant amplitude bit)::

        0 -- 1 -- 2
        |    |    |
        3 -- 4 -- 5
        |    |    |
        6 -- 7 -- 8

    Names and tuple order are stable: X0..X3, Z0..Z3.
    Logical X is the left column; logical Z is the top row.
    """

    n_data = 9
    n_logical = 1
    distance = 3
    x_stabilizers = tuple(
        _pauli("X", support) for support in ((1, 2), (0, 1, 3, 4), (4, 5, 7, 8), (6, 7))
    )
    z_stabilizers = tuple(
        _pauli("Z", support) for support in ((0, 3), (1, 2, 4, 5), (3, 4, 6, 7), (5, 8))
    )
    stabilizers = x_stabilizers + z_stabilizers
    stabilizer_names = ("X0", "X1", "X2", "X3", "Z0", "Z1", "Z2", "Z3")
    logical_x = _pauli("X", (0, 3, 6))
    logical_z = _pauli("Z", (0, 1, 2))
    measurement_orders = (
        (1, 2),
        (0, 1, 3, 4),
        (4, 5, 7, 8),
        (6, 7),
        (0, 3),
        (1, 4, 2, 5),
        (3, 6, 4, 7),
        (5, 8),
    )

    @staticmethod
    def _check_state(state: Statevector) -> None:
        if state.n_qubits != 9:
            raise ValueError("this patch requires exactly nine data qubits")

    def project(self, state: Statevector) -> tuple[Statevector, float]:
        """Return normalized codespace projection and total success probability.

        Does not mutate the input. This ideal projector is a verification and
        state-preparation tool, not a physical preparation protocol.
        Raises ValueError if the input has zero codespace overlap.
        """
        self._check_state(state)
        projected = state.copy()
        probability = 1.0
        for stabilizer in self.stabilizers:
            probability *= projected.postselect(stabilizer)
        return projected, probability

    def logical_zero(self) -> Statevector:
        """Project |000000000> into the +1 stabilizer space; Z_L = +1."""
        state, _ = self.project(Statevector.basis(self.n_data))
        return state

    def logical_one(self) -> Statevector:
        return self.logical_zero().apply_pauli(self.logical_x)

    def encode(self, alpha: complex, beta: complex) -> Statevector:
        """Construct alpha|0_L> + beta|1_L> for normalized coefficients."""
        if (
            not np.isfinite(alpha)
            or not np.isfinite(beta)
            or not np.isclose(abs(alpha) ** 2 + abs(beta) ** 2, 1.0, atol=1e-12, rtol=0)
        ):
            raise ValueError("logical coefficients must be finite and normalized")
        return Statevector(
            alpha * self.logical_zero().amplitudes + beta * self.logical_one().amplitudes
        )

    def logical_state(self, label: str) -> Statevector:
        """Prepare '0', '1', '+', '-', '+i', or '-i' by ideal construction."""
        coefficients = {
            "0": (1, 0),
            "1": (0, 1),
            "+": (1 / np.sqrt(2), 1 / np.sqrt(2)),
            "-": (1 / np.sqrt(2), -1 / np.sqrt(2)),
            "+i": (1 / np.sqrt(2), 1j / np.sqrt(2)),
            "-i": (1 / np.sqrt(2), -1j / np.sqrt(2)),
        }
        if label not in coefficients:
            raise ValueError("logical label must be '0', '1', '+', '-', '+i', or '-i'")
        return self.encode(*coefficients[label])

    def stabilizer_expectations(self, state: Statevector) -> dict[str, float]:
        self._check_state(state)
        return {
            name: state.expectation(stabilizer)
            for name, stabilizer in zip(self.stabilizer_names, self.stabilizers, strict=True)
        }

    def in_codespace(self, state: Statevector, atol: float = 1e-10) -> bool:
        return all(
            np.isclose(value, 1, atol=atol, rtol=0)
            for value in self.stabilizer_expectations(state).values()
        )

    def syndrome(self, state: Statevector, atol: float = 1e-10) -> tuple[int, ...]:
        """Ideal diagnostic bits, 0 for +1 and 1 for -1, in stabilizer order.

        Requires a simultaneous stabilizer eigenstate. A superposition of
        syndromes raises: expectations cannot be interpreted as sampled bits.
        This method does not run a measurement circuit. Use SyndromeExtractorD3
        for sampled ancilla-based extraction, including nondeterministic inputs.
        """
        values = tuple(self.stabilizer_expectations(state).values())
        if not all(np.isclose(abs(v), 1, atol=atol, rtol=0) for v in values):
            raise ValueError("state has no deterministic syndrome")
        return tuple(0 if v > 0 else 1 for v in values)


# Ideal ancilla extraction


@dataclass(frozen=True)
class SyndromeRound:
    """One sampled round, in X0..X3, Z0..Z3 order.

    ``state`` is the collapsed nine-qubit data state, independently owned and
    still mutable. Syndrome bit 1 denotes stabilizer eigenvalue -1.
    Each probability is conditional on the earlier outcomes in this round.
    """

    state: Statevector
    syndrome: tuple[int, ...]
    conditional_probabilities: tuple[float, ...]

    @property
    def probability(self) -> float:
        """Joint probability of this syndrome for the input data state."""
        return prod(self.conditional_probabilities)


class SyndromeExtractorD3:
    """Sequential, ideal extraction on nine data qubits and ancilla q9.

    Checks run in X0..X3, Z0..Z3 order, with horizontal X hooks and vertical
    Z hooks. This class remains a fault-free reference. CircuitExtractorD3
    supplies the explicit gate/reset/idle fault model and audit protocol.
    Only the ancilla is measured; no data-stabilizer projector is used.

    ``extract()`` leaves its input unchanged. For access to the joint state,
    use ``prepare()``, mutating ``measure_check()`` calls, and ``discard_ancilla()``.
    """

    n_data = RotatedSurfaceCodeD3.n_data
    ancilla = n_data
    n_qubits = n_data + 1
    check_names = RotatedSurfaceCodeD3.stabilizer_names
    interaction_orders = RotatedSurfaceCodeD3.measurement_orders
    _ancilla_z = PauliString("I" * n_data + "Z")

    def prepare(self, state: Statevector) -> Statevector:
        """Return |0>_q9 tensor |data>, preserving the data-qubit indices."""
        if state.n_qubits != self.n_data:
            raise ValueError("extraction requires exactly nine data qubits")
        return Statevector(np.concatenate((state.amplitudes, np.zeros(1 << self.n_data))))

    def _check_joint(self, state: Statevector) -> None:
        if state.n_qubits != self.n_qubits:
            raise ValueError("joint state must have ten qubits, with ancilla q9")
        # A reset Z-measured ancilla has an exactly zero |1> block. Reject
        # other inputs rather than silently discarding data/ancilla entanglement.
        if np.any(state.amplitudes[1 << self.n_data :] != 0):
            raise ValueError("ancilla q9 must be reset to |0>")

    def discard_ancilla(self, state: Statevector) -> Statevector:
        """Return an owned data state; requires the ancilla to be reset to |0>."""
        self._check_joint(state)
        return Statevector(state.amplitudes[: 1 << self.n_data])

    def measure_check(
        self,
        state: Statevector,
        name: str,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, float]:
        """Mutate a joint state; return (syndrome bit, outcome probability).

        The input ancilla must be |0>; it is returned to |0> after readout.
        Z checks use data-to-ancilla CNOTs and Z readout. X checks prepare
        the ancilla in |+>, use ancilla-to-data CNOTs, and read out in X
        using a final Hadamard followed by Z measurement.
        """
        self._check_joint(state)
        if not isinstance(name, str) or name not in self.check_names:
            raise ValueError("check name must be X0..X3 or Z0..Z3")
        order = self.interaction_orders[self.check_names.index(name)]
        if name.startswith("X"):
            state.h(self.ancilla)
            for q in order:
                state.cnot(self.ancilla, q)
            state.h(self.ancilla)
        else:
            for q in order:
                state.cnot(q, self.ancilla)

        probabilities = (
            state.probability(self._ancilla_z, 1),
            state.probability(self._ancilla_z, -1),
        )
        bit = state.measure_z(self.ancilla, rng)
        if bit:
            state.x(self.ancilla)
        return bit, probabilities[bit]

    def extract(self, state: Statevector, rng: np.random.Generator | None = None) -> SyndromeRound:
        """Sample all eight checks and return the collapsed data and record.

        Accepts arbitrary normalized nine-qubit data states, including coherent
        superpositions of syndromes. The input is not mutated. There is no
        correction, postselection on a desired syndrome, or noise model.
        """
        joint = self.prepare(state)
        generator = np.random.default_rng() if rng is None else rng
        measurements = [self.measure_check(joint, name, generator) for name in self.check_names]
        return SyndromeRound(
            state=self.discard_ancilla(joint),
            syndrome=tuple(bit for bit, _ in measurements),
            conditional_probabilities=tuple(probability for _, probability in measurements),
        )


# Phenomenological noise


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    try:
        probability = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite probability in [0, 1]") from None
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return probability


@dataclass(frozen=True)
class PauliNoise:
    """Independent per-qubit channel: I, X, Y, Z with probabilities 1-p, px, py, pz.

    Here p = px + py + pz <= 1. One sample is a pure-state trajectory;
    averaging samples represents the stochastic channel, not a coherent rotation.
    """

    px: float = 0.0
    py: float = 0.0
    pz: float = 0.0

    def __post_init__(self) -> None:
        for name in ("px", "py", "pz"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if fsum((self.px, self.py, self.pz)) > 1:
            raise ValueError("px + py + pz must be <= 1")

    @classmethod
    def depolarizing(cls, probability: float) -> "PauliNoise":
        """Total nonidentity probability p, with X/Y/Z each having probability p/3."""
        probability = _probability(probability, "depolarizing probability")
        return cls(probability / 3, probability / 3, probability / 3)

    def sample(self, n_qubits: int, rng: np.random.Generator | None = None) -> PauliString:
        """Sample one Pauli per qubit, with labels in increasing qubit order."""
        n_qubits = _integer(n_qubits, "n_qubits")
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        total = fsum((self.px, self.py, self.pz))
        if total == 0:
            return PauliString("I" * n_qubits)
        generator = np.random.default_rng() if rng is None else rng
        labels = []
        for draw in generator.random(n_qubits):
            if draw < self.px:
                labels.append("X")
            elif draw < self.px + self.py:
                labels.append("Y")
            elif draw < total:
                labels.append("Z")
            else:
                labels.append("I")
        return PauliString("".join(labels))

    def apply(
        self, state: Statevector, rng: np.random.Generator | None = None
    ) -> tuple[Statevector, PauliString]:
        """Return an independently owned noisy state and the sampled Pauli fault."""
        error = self.sample(state.n_qubits, rng)
        return state.copy().apply_pauli(error), error


@dataclass(frozen=True)
class PhenomenologicalNoise:
    """Data noise before each round and independent classical readout-bit flips."""

    data: PauliNoise = field(default_factory=PauliNoise)
    readout: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.data, PauliNoise):
            raise ValueError("data noise must be a PauliNoise instance")
        object.__setattr__(self, "readout", _probability(self.readout, "readout probability"))


@dataclass(frozen=True)
class NoisySyndromeRound:
    """Observed record plus simulator-only truth for validation.

    A decoder should consume ``syndrome`` alone. ``data_error``,
    ``readout_flips``, and ``measurement`` expose otherwise hidden simulation
    truth. The measurement's probabilities are conditioned on the sampled data
    error; they are not the marginal probabilities of the noisy observed record.
    """

    measurement: SyndromeRound
    data_error: PauliString
    readout_flips: tuple[int, ...]

    @property
    def state(self) -> Statevector:
        """Collapsed data state, unaffected by classical readout-bit flips."""
        return self.measurement.state

    @property
    def true_syndrome(self) -> tuple[int, ...]:
        """Ancilla outcomes before classical readout corruption; simulation truth."""
        return self.measurement.syndrome

    @property
    def syndrome(self) -> tuple[int, ...]:
        """Reported syndrome bits, in X0..X3, Z0..Z3 order."""
        return tuple(
            bit ^ flip for bit, flip in zip(self.true_syndrome, self.readout_flips, strict=True)
        )


class NoisySyndromeExtractorD3:
    """Ancilla-based extraction with data and record noise, without correction."""

    def __init__(self, noise: PhenomenologicalNoise | None = None):
        if noise is not None and not isinstance(noise, PhenomenologicalNoise):
            raise ValueError("noise must be a PhenomenologicalNoise instance")
        self.noise = PhenomenologicalNoise() if noise is None else noise
        self._extractor = SyndromeExtractorD3()

    def extract(
        self, state: Statevector, rng: np.random.Generator | None = None
    ) -> NoisySyndromeRound:
        """Sample data errors, extract physically, then corrupt the classical record.

        Leaves the input unchanged. Readout noise is applied after the ideal
        ancilla-reset feedback; it does not model faulty feedback or reset.
        """
        if state.n_qubits != self._extractor.n_data:
            raise ValueError("extraction requires exactly nine data qubits")
        generator = np.random.default_rng() if rng is None else rng
        noisy_state, error = self.noise.data.apply(state, generator)
        measurement = self._extractor.extract(noisy_state, generator)
        if self.noise.readout == 0:
            flips = (0,) * 8
        elif self.noise.readout == 1:
            flips = (1,) * 8
        else:
            flips = tuple(int(draw < self.noise.readout) for draw in generator.random(8))
        return NoisySyndromeRound(measurement, error, flips)

    def run(
        self,
        state: Statevector,
        n_rounds: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[NoisySyndromeRound, ...]:
        """Run repeated rounds, retaining data errors and resampling noise each round.

        The initial input is not mutated. Returned records own independent data
        states. No correction is applied between rounds.
        """
        n_rounds = _integer(n_rounds, "n_rounds")
        if n_rounds < 1:
            raise ValueError("n_rounds must be positive")
        generator = np.random.default_rng() if rng is None else rng
        records = []
        current = state
        for _ in range(n_rounds):
            record = self.extract(current, generator)
            records.append(record)
            current = record.state
        return tuple(records)


# Single-round recovery


def _syndrome_bits(syndrome: Iterable[int]) -> tuple[int, ...]:
    try:
        bits = tuple(syndrome)
    except TypeError:
        raise ValueError("syndrome must contain eight binary integer bits") from None
    if len(bits) != 8 or any(
        not isinstance(bit, (Integral, np.bool_)) or bit not in (0, 1) for bit in bits
    ):
        raise ValueError("syndrome must contain eight binary integer bits")
    return tuple(int(bit) for bit in bits)


def _index(bits: tuple[int, ...]) -> int:
    return sum(bit << q for q, bit in enumerate(bits))


@lru_cache(maxsize=1)
def _recovery_table() -> tuple[PauliString, ...]:
    table: list[PauliString | None] = [None] * 256
    remaining = 256
    for weight in range(10):
        for support in combinations(range(9), weight):
            for axes in product("XYZ", repeat=weight):
                labels = ["I"] * 9
                for q, axis in zip(support, axes, strict=True):
                    labels[q] = axis
                candidate = PauliString("".join(labels))
                bits = tuple(
                    int(not candidate.commutes_with(check))
                    for check in RotatedSurfaceCodeD3.stabilizers
                )
                index = _index(bits)
                if table[index] is None:
                    table[index] = candidate
                    remaining -= 1
                    if remaining == 0:
                        return tuple(entry for entry in table if entry is not None)
    raise RuntimeError("fixed D3 stabilizers do not span all syndromes")


def _check_frame(frame: PauliFrame) -> None:
    if not isinstance(frame, PauliFrame) or frame.n_qubits != 9:
        raise ValueError("this patch requires a nine-qubit PauliFrame")


@dataclass(frozen=True)
class FrameUpdate:
    """Observed bits, bits adjusted for the old frame, and the selected recovery."""

    syndrome: tuple[int, ...]
    effective_syndrome: tuple[int, ...]
    recovery: PauliString
    frame: PauliFrame


class MinimumWeightDecoderD3:
    """Complete 256-entry lookup for ideal, single-round syndrome measurements.

    Minimizes Pauli weight, with deterministic ties: ascending qubit supports,
    then X/Y/Z products. Guarantees recovery of single-qubit Pauli errors, up
    to stabilizers and global phase. Multiple errors may cause logical failure.
    It does not infer measurement faults from time histories or optimize for a
    biased noise model. Inputs are observed bits, never simulator fault labels.
    """

    def __init__(self):
        self._table = _recovery_table()

    def decode(self, syndrome: Iterable[int]) -> PauliString:
        """Return the minimum-weight recovery for the supplied eight-bit syndrome."""
        return self._table[_index(_syndrome_bits(syndrome))]

    def correct(self, state: Statevector, syndrome: Iterable[int]) -> Statevector:
        """Explicit correction reference for data with an identity starting frame."""
        if state.n_qubits != 9:
            raise ValueError("this patch requires exactly nine data qubits")
        return state.copy().apply_pauli(self.decode(syndrome))

    def update(self, frame: PauliFrame, syndrome: Iterable[int]) -> FrameUpdate:
        """Decode in the current corrected frame and return a new frame.

        effective = observed XOR syndrome(old_frame)
        new_frame = recovery(effective) * old_frame, modulo global phase.
        Repeating an unchanged ideal syndrome therefore adds no new recovery.
        """
        _check_frame(frame)
        observed = _syndrome_bits(syndrome)
        effective = tuple(
            bit ^ ((1 - frame.sign(check)) // 2)
            for bit, check in zip(observed, RotatedSurfaceCodeD3.stabilizers, strict=True)
        )
        recovery = self.decode(effective)
        return FrameUpdate(observed, effective, recovery, frame.updated(recovery))


@dataclass(frozen=True)
class FrameCorrectionRound:
    """Raw measured data plus an immutable correction frame and decoder record."""

    measurement: NoisySyndromeRound
    update: FrameUpdate

    @property
    def state(self) -> Statevector:
        """Raw data; the pending Pauli recovery has not been applied to it."""
        return self.measurement.state

    @property
    def frame(self) -> PauliFrame:
        return self.update.frame

    @property
    def corrected_state(self) -> Statevector:
        """Materialized corrected copy for validation or export; raw data is unchanged."""
        return self.frame.apply(self.state)


class PauliFrameCorrectionD3:
    """Extract, decode, and update a frame without applying data correction gates.

    Supports pre-round Pauli noise with perfect syndrome readout. Readout noise
    is rejected because this decoder does not use a time history. Use
    HistoryPauliFrameCorrectionD3 for noisy readout. Raw data and
    frame must be carried together between steps.
    """

    def __init__(self, noise: PhenomenologicalNoise | None = None):
        self._extractor = NoisySyndromeExtractorD3(noise)
        if self._extractor.noise.readout != 0:
            raise ValueError(
                "readout noise needs a history decoder; use HistoryPauliFrameCorrectionD3"
            )
        self.decoder = MinimumWeightDecoderD3()

    def step(
        self,
        state: Statevector,
        frame: PauliFrame | None = None,
        rng: np.random.Generator | None = None,
    ) -> FrameCorrectionRound:
        """Return measured raw data and an updated frame; do not mutate the input."""
        previous = PauliFrame(9) if frame is None else frame
        _check_frame(previous)
        measurement = self._extractor.extract(state, rng)
        return FrameCorrectionRound(
            measurement, self.decoder.update(previous, measurement.syndrome)
        )

    def run(
        self,
        state: Statevector,
        n_rounds: int,
        frame: PauliFrame | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[FrameCorrectionRound, ...]:
        """Carry raw data and its correction frame through repeated extraction rounds."""
        n_rounds = _integer(n_rounds, "n_rounds")
        if n_rounds < 1:
            raise ValueError("n_rounds must be positive")
        generator = np.random.default_rng() if rng is None else rng
        records = []
        current, current_frame = state, frame
        for _ in range(n_rounds):
            record = self.step(current, current_frame, generator)
            records.append(record)
            current, current_frame = record.state, record.frame
        return tuple(records)


# Phenomenological history decoding

_HISTORY_STATES = np.arange(1024)
_HAMMING = np.array([bits.bit_count() for bits in range(256)])
_CHECKS = (
    *RotatedSurfaceCodeD3.stabilizers,
    RotatedSurfaceCodeD3.logical_z,
    RotatedSurfaceCodeD3.logical_x,
)


def _class_index(frame: PauliFrame) -> int:
    return sum(((1 - frame.sign(check)) // 2) << bit for bit, check in enumerate(_CHECKS))


@lru_cache(maxsize=1)
def _representatives() -> tuple[PauliFrame, ...]:
    decoder = MinimumWeightDecoderD3()
    frames = []
    for target in range(1024):
        frame = PauliFrame.from_pauli(
            decoder.decode(tuple((target >> bit) & 1 for bit in range(8)))
        )
        difference = _class_index(frame) ^ target
        if difference & 256:
            frame = frame.updated(RotatedSurfaceCodeD3.logical_x)
        if difference & 512:
            frame = frame.updated(RotatedSurfaceCodeD3.logical_z)
        frames.append(frame)
    return tuple(frames)


@lru_cache(maxsize=1)
def _transition_indices() -> NDArray[np.int64]:
    indices = np.empty((9, 3, 1024), dtype=np.int64)
    for q in range(9):
        for axis_index, axis in enumerate("XYZ"):
            error = PauliString("I" * q + axis + "I" * (8 - q))
            shift = _class_index(PauliFrame.from_pauli(error))
            indices[q, axis_index] = _HISTORY_STATES ^ shift
    indices.flags.writeable = False
    return indices


@dataclass(frozen=True)
class HistoryFrameUpdate:
    """A posterior-MAP recovery decision after one observed record.

    ``confidence`` is the posterior probability of the selected stabilizer
    class under the configured model, not a guarantee of recovery. Class ties
    select the lowest index. ``final_readout`` marks explicit noiseless closure.
    """

    syndrome: tuple[int, ...]
    estimated_syndrome: tuple[int, ...]
    recovery: PauliString
    frame: PauliFrame
    confidence: float
    log_evidence: float
    n_rounds: int
    final_readout: bool = False


class HistoryDecoderD3:
    """Streaming posterior over all Pauli/stabilizer classes for a known encoding.

    Each update applies independent data noise, then conditions on eight noisy
    observed bits. The initial raw state is assumed to differ from the intended
    encoded state only by ``initial_frame``. State amplitudes and hidden simulator
    fault labels are never inputs. Reuse this instance to retain uncertainty
    across a stream; a MAP frame alone does not retain the posterior.
    """

    def __init__(
        self,
        noise: PhenomenologicalNoise | None = None,
        initial_frame: PauliFrame | None = None,
    ):
        if noise is not None and not isinstance(noise, PhenomenologicalNoise):
            raise ValueError("noise must be a PhenomenologicalNoise instance")
        self._noise = PhenomenologicalNoise() if noise is None else noise
        self._frames = _representatives()
        self.reset(initial_frame)

    @property
    def noise(self) -> PhenomenologicalNoise:
        return self._noise

    @property
    def posterior(self) -> NDArray[np.float64]:
        """Owned copy; index bits 0..7 are syndrome, bits 8/9 anticommute with ZL/XL."""
        return self._posterior.copy()

    @property
    def frame(self) -> PauliFrame:
        return self._frame

    def representative(self, index: int) -> PauliFrame:
        """Return a correction representative for an index in ``posterior``."""
        index = _integer(index, "class index")
        if not 0 <= index < 1024:
            raise ValueError("class index must be in [0, 1024)")
        return self._frames[index]

    @property
    def n_rounds(self) -> int:
        return self._n_rounds

    @property
    def log_likelihood(self) -> float:
        return self._log_likelihood

    def reset(self, initial_frame: PauliFrame | None = None) -> None:
        """Start an independent history with a known initial frame, not an estimate."""
        frame = PauliFrame(9) if initial_frame is None else initial_frame
        _check_frame(frame)
        posterior = np.zeros(1024)
        posterior[_class_index(frame)] = 1
        self._posterior = posterior
        self._frame = frame
        self._n_rounds = 0
        self._log_likelihood = 0.0
        self._finished = False

    def _predict(self) -> NDArray[np.float64]:
        channel = self.noise.data
        p = fsum((channel.px, channel.py, channel.pz))
        prior = self._posterior.copy()
        if p == 0:
            return prior
        for indices in _transition_indices():
            previous = prior
            prior = (
                (1 - p) * previous
                + channel.px * previous[indices[0]]
                + channel.py * previous[indices[1]]
                + channel.pz * previous[indices[2]]
            )
        return prior

    def _condition(
        self,
        prior: NDArray[np.float64],
        observed: tuple[int, ...],
        readout: float,
        final: bool,
    ) -> HistoryFrameUpdate:
        distances = _HAMMING[(_HISTORY_STATES & 255) ^ _index(observed)]
        log_weights = np.full(1024, -np.inf)
        positive = prior > 0
        if readout in (0, 1):
            positive &= distances == (0 if readout == 0 else 8)
            log_weights[positive] = np.log(prior[positive])
        else:
            log_weights[positive] = (
                np.log(prior[positive])
                + distances[positive] * np.log(readout)
                + (8 - distances[positive]) * np.log1p(-readout)
            )
        peak = float(np.max(log_weights))
        if not np.isfinite(peak):
            raise ValueError("zero-probability syndrome history under the configured noise model")
        weights = np.exp(log_weights - peak)
        total = float(weights.sum())
        posterior = weights / total
        best = int(np.argmax(posterior))
        # Preserve the exact existing representative if its stabilizer class wins.
        frame = self._frame if _class_index(self._frame) == best else self._frames[best]
        recovery = self._frame.updated(frame.pauli).pauli
        log_evidence = peak + float(np.log(total))
        rounds = self._n_rounds + int(not final)
        result = HistoryFrameUpdate(
            syndrome=observed,
            estimated_syndrome=tuple((best >> bit) & 1 for bit in range(8)),
            recovery=recovery,
            frame=frame,
            confidence=float(posterior[best]),
            log_evidence=log_evidence,
            n_rounds=rounds,
            final_readout=final,
        )
        self._posterior = posterior
        self._frame = frame
        self._n_rounds = rounds
        self._log_likelihood += log_evidence
        self._finished = final
        return result

    def update(self, syndrome: Iterable[int]) -> HistoryFrameUpdate:
        """Consume one noisy round: data transition followed by record likelihood."""
        if self._finished:
            raise ValueError("history is finished; reset before adding more rounds")
        observed = _syndrome_bits(syndrome)
        return self._condition(self._predict(), observed, self.noise.readout, False)

    def finish(self, syndrome: Iterable[int]) -> HistoryFrameUpdate:
        """Condition on an explicit perfect final measurement, with no new data noise.

        This is an additional experimental assumption, not access to simulator
        truth. The correction runner obtains it by executing another ancilla round.
        """
        if self._finished:
            raise ValueError("history is already finished")
        observed = _syndrome_bits(syndrome)
        return self._condition(self._posterior, observed, 0.0, True)

    def decode(
        self,
        history: Iterable[Iterable[int]],
        initial_frame: PauliFrame | None = None,
        final_syndrome: Iterable[int] | None = None,
    ) -> HistoryFrameUpdate:
        """Decode an independent history without mutating this streaming instance."""
        decoder = HistoryDecoderD3(self.noise, initial_frame)
        try:
            observations = iter(history)
        except TypeError:
            raise ValueError("history must contain noisy syndrome records") from None
        result = None
        for observed in observations:
            result = decoder.update(observed)
        if result is None:
            raise ValueError("history must contain at least one noisy round")
        return result if final_syndrome is None else decoder.finish(final_syndrome)


@dataclass(frozen=True)
class HistoryCorrectionRound:
    measurement: NoisySyndromeRound
    update: HistoryFrameUpdate

    @property
    def state(self) -> Statevector:
        return self.measurement.state

    @property
    def frame(self) -> PauliFrame:
        return self.update.frame

    @property
    def corrected_state(self) -> Statevector:
        return self.frame.apply(self.state)


@dataclass(frozen=True)
class HistoryCorrectionResult:
    """Independent storage experiment, with an optional actual perfect end measurement."""

    rounds: tuple[HistoryCorrectionRound, ...]
    final_measurement: SyndromeRound | None = None
    final_update: HistoryFrameUpdate | None = None

    @property
    def state(self) -> Statevector:
        return (
            self.rounds[-1].state
            if self.final_measurement is None
            else self.final_measurement.state
        )

    @property
    def update(self) -> HistoryFrameUpdate:
        return self.rounds[-1].update if self.final_update is None else self.final_update

    @property
    def frame(self) -> PauliFrame:
        return self.update.frame

    @property
    def corrected_state(self) -> Statevector:
        return self.frame.apply(self.state)


class HistoryPauliFrameCorrectionD3:
    """Run noisy memory rounds and Bayesian frame updates without data recovery gates.

    Initial data must be encoded up to a known initial frame. Each run starts
    a new posterior; for continuous filtering, retain a HistoryDecoderD3 instance
    alongside a NoisySyndromeExtractorD3 instead of restarting from a MAP frame.
    """

    def __init__(self, noise: PhenomenologicalNoise | None = None):
        self._extractor = NoisySyndromeExtractorD3(noise)

    def run(
        self,
        state: Statevector,
        n_rounds: int,
        initial_frame: PauliFrame | None = None,
        rng: np.random.Generator | None = None,
        *,
        final_readout: bool = False,
    ) -> HistoryCorrectionResult:
        n_rounds = _integer(n_rounds, "n_rounds")
        if n_rounds < 1:
            raise ValueError("n_rounds must be positive")
        if not isinstance(final_readout, bool):
            raise ValueError("final_readout must be a Boolean")
        if state.n_qubits != 9:
            raise ValueError("this patch requires exactly nine data qubits")
        decoder = HistoryDecoderD3(self._extractor.noise, initial_frame)
        generator = np.random.default_rng() if rng is None else rng
        current = state
        records = []
        for _ in range(n_rounds):
            measured = self._extractor.extract(current, generator)
            update = decoder.update(measured.syndrome)
            records.append(HistoryCorrectionRound(measured, update))
            current = measured.state
        if final_readout:
            measured_final = SyndromeExtractorD3().extract(current, generator)
            update_final = decoder.finish(measured_final.syndrome)
            return HistoryCorrectionResult(tuple(records), measured_final, update_final)
        return HistoryCorrectionResult(tuple(records))


# Circuit faults and extraction

_SINGLE = ("X", "Y", "Z")
_PAIR = tuple("".join(p) for p in product("IXYZ", repeat=2) if p != ("I", "I"))
_SAFE_ORDERS = RotatedSurfaceCodeD3.measurement_orders


@dataclass(frozen=True)
class CircuitNoise:
    """Per-location event probabilities; Pauli alternatives are mutually exclusive.

    single_qubit: total X/Y/Z probability after H; two_qubit: total probability
    of the 15 nonidentity Paulis after CNOT. Preparation/reset are X faults on
    the ancilla. Measurement flips its reported bit before feedback. Idle is
    total X/Y/Z probability per inactive qubit per tick. All defaults are zero.
    """

    single_qubit: float = 0
    two_qubit: float = 0
    preparation: float = 0
    reset: float = 0
    measurement: float = 0
    idle: float = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _probability(getattr(self, name), name))

    @classmethod
    def uniform(cls, probability: float) -> "CircuitNoise":
        return cls(*([probability] * 6))


@dataclass(frozen=True)
class CircuitScheduleD3:
    """X hooks horizontal, Z hooks vertical; legacy ascending order is retained for audits."""

    orders: tuple[tuple[int, ...], ...] = _SAFE_ORDERS

    def __post_init__(self) -> None:
        try:
            orders = tuple(tuple(_integer(q, "qubit") for q in order) for order in self.orders)
        except TypeError:
            raise ValueError("orders must contain eight support permutations") from None
        checks = RotatedSurfaceCodeD3.stabilizers
        if len(orders) != 8 or any(
            sorted(order) != list(check.support)
            for order, check in zip(orders, checks, strict=True)
        ):
            raise ValueError("orders must contain eight support permutations")
        object.__setattr__(self, "orders", orders)

    @classmethod
    def legacy(cls) -> "CircuitScheduleD3":
        return cls(tuple(check.support for check in RotatedSurfaceCodeD3.stabilizers))


DEFAULT_SCHEDULE = CircuitScheduleD3()


@dataclass(frozen=True)
class FaultLocation:
    name: str
    rate: str
    targets: tuple[int, ...]
    alternatives: tuple[str, ...]


@dataclass(frozen=True)
class CircuitOperation:
    name: str
    gate: str
    targets: tuple[int, ...]
    check: int
    locations: tuple[FaultLocation, ...]


def _operation(name: str, gate: str, targets: tuple[int, ...], check: int) -> CircuitOperation:
    rate, alternatives = {
        "prepare": ("preparation", ("X",)),
        "h": ("single_qubit", _SINGLE),
        "cnot": ("two_qubit", _PAIR),
        "measure": ("measurement", ("flip",)),
        "feedback": ("reset", ("X",)),
    }[gate]
    locations = [FaultLocation(name + ".fault", rate, targets, alternatives)]
    locations += [
        FaultLocation(f"{name}.idle.q{q}", "idle", (q,), _SINGLE)
        for q in range(10)
        if q not in targets
    ]
    return CircuitOperation(name, gate, targets, check, tuple(locations))


PREPARATION = _operation("prepare", "prepare", (9,), -1)


@lru_cache(maxsize=32)
def circuit_operations(
    schedule: CircuitScheduleD3 = DEFAULT_SCHEDULE,
) -> tuple[CircuitOperation, ...]:
    operations = []
    for check, (name, order) in enumerate(
        zip(RotatedSurfaceCodeD3.stabilizer_names, schedule.orders, strict=True)
    ):
        if name.startswith("X"):
            operations.append(_operation(name + ".h_prepare", "h", (9,), check))
        for position, q in enumerate(order):
            targets = (9, q) if name.startswith("X") else (q, 9)
            operations.append(_operation(f"{name}.cx.{position}", "cnot", targets, check))
        if name.startswith("X"):
            operations.append(_operation(name + ".h_readout", "h", (9,), check))
        operations.append(_operation(name + ".measure", "measure", (9,), check))
        operations.append(_operation(name + ".feedback", "feedback", (9,), check))
    return tuple(operations)


@dataclass(frozen=True)
class CircuitFault:
    """A forced event; round -1 denotes the one initial preparation tick."""

    round_index: int
    location: str
    pauli: str


def fault_locations(
    schedule: CircuitScheduleD3 = DEFAULT_SCHEDULE, *, preparation: bool = False
) -> tuple[FaultLocation, ...]:
    operations = (PREPARATION,) if preparation else circuit_operations(schedule)
    return tuple(location for operation in operations for location in operation.locations)


def _injections(faults: Iterable[CircuitFault], n_rounds: int, schedule: CircuitScheduleD3) -> dict:
    locations = {location.name: location for location in fault_locations(schedule)}
    prep = {location.name: location for location in PREPARATION.locations}
    result = {}
    for fault in faults:
        if not isinstance(fault, CircuitFault):
            raise ValueError("injections must be CircuitFault instances")
        if not isinstance(fault.location, str) or not isinstance(fault.pauli, str):
            raise ValueError("fault location and Pauli alternative must be strings")
        stage = _integer(fault.round_index, "round_index")
        if not -1 <= stage < n_rounds:
            raise ValueError("fault round is outside the experiment")
        location = (prep if stage == -1 else locations).get(fault.location)
        if location is None or fault.pauli not in location.alternatives:
            raise ValueError("unknown fault location or invalid Pauli alternative")
        key = (stage, fault.location)
        if key in result:
            raise ValueError("only one forced alternative is allowed at a location")
        result[key] = fault
    return result


def _local_masks(targets: tuple[int, ...], labels: str) -> tuple[int, int]:
    x = sum((axis in "XY") << q for q, axis in zip(targets, labels, strict=True))
    z = sum((axis in "YZ") << q for q, axis in zip(targets, labels, strict=True))
    return x, z


def _data_state(joint: Statevector) -> Statevector:
    vector = joint.amplitudes
    low, high = vector[:512], vector[512:]
    if not np.any(high != 0):
        return Statevector(low)
    if not np.any(low != 0):
        return Statevector(high)
    raise ValueError("ancilla must have undergone computational-basis measurement")


@dataclass(frozen=True)
class CircuitPreparation:
    """Initial ten-qubit state and preparation diagnostics, independently owned."""

    joint_state: Statevector
    faults: tuple[CircuitFault, ...]


@dataclass(frozen=True)
class CircuitRound:
    state: Statevector
    joint_state: Statevector
    syndrome: tuple[int, ...]
    true_syndrome: tuple[int, ...]
    faults: tuple[CircuitFault, ...]


@dataclass(frozen=True)
class CircuitMemoryResult:
    rounds: tuple[CircuitRound, ...]
    preparation_faults: tuple[CircuitFault, ...]
    final_measurement: SyndromeRound | None = None

    @property
    def state(self) -> Statevector:
        return (
            self.rounds[-1].state
            if self.final_measurement is None
            else self.final_measurement.state
        )


def clean_readout(joint: Statevector, rng: np.random.Generator) -> SyndromeRound:
    """Actual ideal ancilla reset and syndrome extraction, without new data faults."""
    current = joint.copy()
    if current.measure_z(9, rng):
        current.x(9)
    extractor = SyndromeExtractorD3()
    measured = [extractor.measure_check(current, name, rng) for name in extractor.check_names]
    return SyndromeRound(
        _data_state(current), tuple(bit for bit, _ in measured), tuple(p for _, p in measured)
    )


class CircuitExtractorD3:
    """Dense-statevector circuit with sampled or forced faults and reported-bit feedback.

    Forced events override the random choice at that location. Use zero rates
    for a deterministic single-fault experiment. Preparation occurs once per
    memory experiment; the same ancilla is carried between checks and rounds.
    """

    def __init__(
        self, noise: CircuitNoise | None = None, schedule: CircuitScheduleD3 | None = None
    ):
        if noise is not None and not isinstance(noise, CircuitNoise):
            raise ValueError("noise must be CircuitNoise")
        if schedule is not None and not isinstance(schedule, CircuitScheduleD3):
            raise ValueError("schedule must be CircuitScheduleD3")
        self.noise = CircuitNoise() if noise is None else noise
        self.schedule = CircuitScheduleD3() if schedule is None else schedule

    def _faults(self, state, operation, stage, forced, rng, observed, sample_noise=True):
        events = []
        for location in operation.locations:
            event = forced.get((stage, location.name))
            if event is None:
                p = getattr(self.noise, location.rate) if sample_noise else 0
                if p == 0 or rng.random() >= p:
                    continue
                label = location.alternatives[int(rng.integers(len(location.alternatives)))]
                event = CircuitFault(stage, location.name, label)
            events.append(event)
            if event.pauli == "flip":
                observed ^= 1
            else:
                x, z = _local_masks(location.targets, event.pauli)
                state.apply_pauli(PauliFrame(10, x, z).pauli)
        return observed, events

    def prepare(
        self, state: Statevector, rng=None, *, injections=(), sample_noise=True
    ) -> CircuitPreparation:
        """Prepare once, then retain joint_state between extract_round calls.

        Only round -1 injections belong here. The ancilla is not reset again
        when a subsequent round starts, including after faulty feedback.
        """
        if not isinstance(sample_noise, bool):
            raise ValueError("sample_noise must be Boolean")
        forced = _injections(injections, 1, self.schedule)
        if any(stage != -1 for stage, _ in forced):
            raise ValueError("preparation injections must use round -1")
        generator = np.random.default_rng() if rng is None else rng
        joint = SyndromeExtractorD3().prepare(state)
        _, events = self._faults(joint, PREPARATION, -1, forced, generator, 0, sample_noise)
        return CircuitPreparation(joint, tuple(events))

    def _execute_round(self, joint, stage, generator, forced, sample_noise):
        bits, true_bits, events = [], [], []
        observed = 0
        for operation in circuit_operations(self.schedule):
            if operation.gate == "h":
                joint.h(*operation.targets)
            elif operation.gate == "cnot":
                joint.cnot(*operation.targets)
            elif operation.gate == "measure":
                observed = joint.measure_z(9, generator)
                true_bits.append(observed)
            elif operation.gate == "feedback" and observed:
                joint.x(9)
            observed, faults = self._faults(
                joint, operation, stage, forced, generator, observed, sample_noise
            )
            events.extend(faults)
            if operation.gate == "measure":
                bits.append(observed)
        return CircuitRound(
            _data_state(joint), joint.copy(), tuple(bits), tuple(true_bits), tuple(events)
        )

    def extract_round(
        self,
        joint_state: Statevector,
        rng=None,
        *,
        round_index=0,
        injections=(),
        sample_noise=True,
    ) -> CircuitRound:
        """One nonmutating round on the existing physical ten-qubit state.

        Carries a residual ancilla |1> through the next check. This low-level
        interface does not impose an encoded-input prior or run a decoder.
        Fault indices are absolute within the continuing memory experiment.
        """
        stage = _integer(round_index, "round_index")
        if stage < 0 or joint_state.n_qubits != 10:
            raise ValueError("nonnegative round_index and a ten-qubit joint state are required")
        if not isinstance(sample_noise, bool):
            raise ValueError("sample_noise must be Boolean")
        forced = _injections(injections, stage + 1, self.schedule)
        if any(index != stage for index, _ in forced):
            raise ValueError("round injections must match round_index")
        generator = np.random.default_rng() if rng is None else rng
        return self._execute_round(joint_state.copy(), stage, generator, forced, sample_noise)

    def run(
        self,
        state: Statevector,
        n_rounds: int,
        rng: np.random.Generator | None = None,
        *,
        injections: Iterable[CircuitFault] = (),
        final_readout: bool = False,
        sample_noise: bool = True,
    ) -> CircuitMemoryResult:
        n_rounds = _integer(n_rounds, "n_rounds")
        if n_rounds < 1 or state.n_qubits != 9:
            raise ValueError("positive n_rounds and exactly nine data qubits are required")
        if not isinstance(final_readout, bool):
            raise ValueError("final_readout must be Boolean")
        if not isinstance(sample_noise, bool):
            raise ValueError("sample_noise must be Boolean")
        forced = _injections(injections, n_rounds, self.schedule)
        generator = np.random.default_rng() if rng is None else rng
        joint = SyndromeExtractorD3().prepare(state)
        _, preparation = self._faults(joint, PREPARATION, -1, forced, generator, 0, sample_noise)
        records = []
        for stage in range(n_rounds):
            records.append(self._execute_round(joint, stage, generator, forced, sample_noise))
        final = clean_readout(joint, generator) if final_readout else None
        return CircuitMemoryResult(tuple(records), tuple(preparation), final)


@dataclass(frozen=True)
class FaultTrace:
    """Independent binary-Pauli trajectory for auditing encoded inputs."""

    history: tuple[tuple[int, ...], ...]
    true_history: tuple[tuple[int, ...], ...]
    data_errors: tuple[PauliFrame, ...]
    ancilla_bits: tuple[int, ...]

    @property
    def final_error(self) -> PauliFrame:
        return self.data_errors[-1]

    @property
    def final_syndrome(self) -> tuple[int, ...]:
        value = _class_index(self.final_error)
        return tuple((value >> bit) & 1 for bit in range(8))

    @property
    def signature(self) -> tuple[tuple[int, ...], ...]:
        return (*self.history, self.final_syndrome)


def trace_faults(
    faults: Iterable[CircuitFault],
    n_rounds: int = 3,
    schedule: CircuitScheduleD3 = DEFAULT_SCHEDULE,
    input_error: PauliFrame | None = None,
) -> FaultTrace:
    """Propagate full physical masks, independently of the dense circuit and decoder."""
    n_rounds = _integer(n_rounds, "n_rounds")
    if n_rounds < 1:
        raise ValueError("n_rounds must be positive")
    forced = _injections(faults, n_rounds, schedule)
    initial = PauliFrame(9) if input_error is None else input_error
    if not isinstance(initial, PauliFrame) or initial.n_qubits != 9:
        raise ValueError("input_error must have nine qubits")
    x, z = initial.x_mask, initial.z_mask
    history, true_history, data, ancillas = [], [], [], []
    for stage, operations in [
        (-1, (PREPARATION,)),
        *((t, circuit_operations(schedule)) for t in range(n_rounds)),
    ]:
        bits, true_bits = [], []
        observed = 0
        for operation in operations:
            if operation.gate == "h":
                a = operation.targets[0]
                difference = ((x >> a) ^ (z >> a)) & 1
                x ^= difference << a
                z ^= difference << a
            elif operation.gate == "cnot":
                a, b = operation.targets
                x ^= ((x >> a) & 1) << b
                z ^= ((z >> b) & 1) << a
            elif operation.gate == "measure":
                observed = (x >> 9) & 1
                true_bits.append(observed)
                z &= 511
            elif operation.gate == "feedback":
                x ^= observed << 9
            for location in operation.locations:
                fault = forced.get((stage, location.name))
                if fault is None:
                    continue
                if fault.pauli == "flip":
                    observed ^= 1
                else:
                    dx, dz = _local_masks(location.targets, fault.pauli)
                    x ^= dx
                    z ^= dz
            if operation.gate == "measure":
                bits.append(observed)
        if stage >= 0:
            history.append(tuple(bits))
            true_history.append(tuple(true_bits))
            data.append(PauliFrame(9, x & 511, z & 511))
            ancillas.append((x >> 9) & 1)
    return FaultTrace(tuple(history), tuple(true_history), tuple(data), tuple(ancillas))


# Correlated circuit history decoding

_STATES = np.arange(2048)
_DATA = _STATES & 1023
_STANDARD = tuple(
    PauliFrame.from_pauli(PauliString(check.labels + "I"))
    for check in (
        *RotatedSurfaceCodeD3.stabilizers,
        RotatedSurfaceCodeD3.logical_z,
        RotatedSurfaceCodeD3.logical_x,
    )
) + (PauliFrame(10, 0, 512),)


def _shift(x, z, basis):
    return sum(
        (((x & check.z_mask).bit_count() + (z & check.x_mask).bit_count()) % 2) << bit
        for bit, check in enumerate(basis)
    )


def _channels(locations, basis):
    return tuple(
        (
            location.rate,
            tuple(
                _shift(*_local_masks(location.targets, axis), basis)
                for axis in location.alternatives
            ),
        )
        for location in locations
        if location.rate != "measurement"
    )


@lru_cache(maxsize=32)
def _compiled_channels(schedule):
    checks, pre = [], []
    basis = _STANDARD
    for operation in circuit_operations(schedule):
        if operation.gate in ("h", "cnot"):
            basis = tuple(check.propagated(operation.gate, *operation.targets) for check in basis)
            pre.extend(_channels(operation.locations, basis))
        elif operation.gate == "measure":
            # The measured ancilla Z is G_ancilla * S_check in this basis.
            # Conditioning and rebasing restore the standard basis below.
            post = list(_channels(operation.locations, _STANDARD))
        elif operation.gate == "feedback":
            post.extend(_channels(operation.locations, _STANDARD))
            checks.append((tuple(pre), tuple(post)))
            pre, basis = [], _STANDARD
    return _channels(PREPARATION.locations, _STANDARD), tuple(checks)


def _fwht(values):
    result = np.array(values, dtype=float, copy=True)
    width = 1
    while width < result.size:
        blocks = result.reshape(-1, 2 * width)
        left, right = blocks[:, :width].copy(), blocks[:, width:].copy()
        blocks[:, :width] = left + right
        blocks[:, width:] = left - right
        width *= 2
    return result


class _Kernel:
    """All-order interval channel; zero-probability support is retained exactly."""

    def __init__(self, channels, noise):
        distribution = np.zeros(2048)
        distribution[0] = 1
        for rate, shifts in channels:
            p = getattr(noise, rate)
            if p == 0:
                continue
            previous = distribution
            distribution = (1 - p) * previous
            for shift in shifts:
                distribution = distribution + (p / len(shifts)) * previous[_STATES ^ shift]
        distribution /= distribution.sum()
        self.distribution = distribution
        self.support = np.flatnonzero(distribution)
        self.transform = _fwht(distribution)
        self.support_transform = _fwht(distribution > 0)

    def apply(self, prior):
        if len(self.support) <= 32:
            result = np.zeros(2048)
            for shift in self.support:
                result += self.distribution[shift] * prior[_STATES ^ shift]
        else:
            result = _fwht(_fwht(prior) * self.transform) / 2048
            if float(result.min()) < -1e-12:
                raise ArithmeticError("negative circuit-channel probability beyond roundoff")
            result = np.maximum(result, 0)
            if len(self.support) < 2048:
                # Boolean support convolution has integer counts <=2048; its
                # Walsh arithmetic is exact in float64, unlike tiny probabilities.
                counts = _fwht(_fwht(prior > 0) * self.support_transform) / 2048
                result[np.rint(counts) == 0] = 0
        total = float(result.sum())
        if not total > 0:
            raise ArithmeticError("circuit-channel probability mass vanished")
        return result / total


@lru_cache(maxsize=32)
def _kernels(noise, schedule):
    preparation, checks = _compiled_channels(schedule)
    return _Kernel(preparation, noise), tuple(
        (_Kernel(pre, noise), _Kernel(post, noise)) for pre, post in checks
    )


@dataclass(frozen=True)
class CircuitFrameUpdate(HistoryFrameUpdate):
    """A certificate assumes <=1 fault; catalogue compatibility is not fault truth.

    Confidence remains the selected class's unmodified posterior probability.
    """

    single_fault_certified: bool = False
    map_confidence: float = 0.0


class CircuitHistoryDecoderD3:
    """Bayesian memory decoder matched to CircuitExtractorD3, with noisy feedback.

    Joint posterior index: low ten bits are the data Pauli/stabilizer class;
    bit ten is the ancilla X error in the check-boundary reference state. A
    transported basis is used within checks. Ancilla states are marginalized
    before selecting a data recovery. All fault combinations are included.
    At a matching clean boundary, certification_rounds enables the unambiguous
    <=1-fault catalogue policy; other decisions use MAP. Set it to None for
    pure MAP. Confidence always remains the selected class's posterior mass.
    """

    def __init__(
        self,
        noise: CircuitNoise | None = None,
        schedule: CircuitScheduleD3 | None = None,
        initial_frame: PauliFrame | None = None,
        *,
        certification_rounds: int | None = 3,
    ):
        extractor = CircuitExtractorD3(noise, schedule)
        self._noise, self._schedule = extractor.noise, extractor.schedule
        self._preparation, self._checks = _kernels(self.noise, self.schedule)
        self._frames = _representatives()
        if certification_rounds is not None:
            certification_rounds = _integer(certification_rounds, "certification_rounds")
            if certification_rounds < 1:
                raise ValueError("certification_rounds must be positive or None")
        self.certification_rounds = certification_rounds
        self.reset(initial_frame)

    @property
    def noise(self):
        return self._noise

    @property
    def schedule(self):
        return self._schedule

    @property
    def posterior(self):
        return self._posterior.copy()

    @property
    def data_posterior(self):
        return self._posterior.reshape(2, 1024).sum(axis=0)

    @property
    def frame(self):
        return self._frame

    @property
    def n_rounds(self):
        return self._n_rounds

    @property
    def log_likelihood(self):
        return self._log_likelihood

    def reset(self, initial_frame=None):
        frame = PauliFrame(9) if initial_frame is None else initial_frame
        _check_frame(frame)
        prior = np.zeros(2048)
        prior[_class_index(frame)] = 1
        self._posterior = self._preparation.apply(prior)
        self._frame, self._n_rounds, self._log_likelihood, self._finished = frame, 0, 0.0, False
        self._initial_frame = frame
        self._records = []

    def _decision(
        self, posterior, observed, log_evidence, final, selected_class=None, certified=False
    ):
        marginal = posterior.reshape(2, 1024).sum(axis=0)
        best = int(np.argmax(marginal)) if selected_class is None else selected_class
        frame = self._frame if _class_index(self._frame) == best else self._frames[best]
        result = CircuitFrameUpdate(
            observed,
            tuple((best >> bit) & 1 for bit in range(8)),
            self._frame.updated(frame.pauli).pauli,
            frame,
            float(marginal[best]),
            log_evidence,
            self._n_rounds + int(not final),
            final,
            single_fault_certified=certified,
            map_confidence=float(marginal.max()),
        )
        self._posterior, self._frame = posterior, frame
        self._n_rounds = result.n_rounds
        self._log_likelihood += log_evidence
        self._finished = final
        return result

    def update(self, syndrome: Iterable[int]) -> CircuitFrameUpdate:
        if self._finished:
            raise ValueError("history is finished; reset before adding rounds")
        observed = _syndrome_bits(syndrome)
        posterior = self._posterior.copy()
        evidence = 0.0
        q = self.noise.measurement
        for check, (bit, (pre, post)) in enumerate(zip(observed, self._checks, strict=True)):
            prior = pre.apply(posterior)
            true = ((_STATES >> 10) ^ (_STATES >> check)) & 1
            likelihood = np.where(true == bit, 1 - q, q)
            positive = (prior > 0) & (likelihood > 0)
            log_weights = np.full(2048, -np.inf)
            log_weights[positive] = np.log(prior[positive]) + np.log(likelihood[positive])
            peak = float(np.max(log_weights))
            if not np.isfinite(peak):
                raise ValueError("zero-probability circuit history under configured noise")
            weights = np.exp(log_weights - peak)
            total = float(weights.sum())
            # Z measurement then X**reported feedback. The outgoing physical
            # ancilla bit is true XOR reported, correlated with record corruption.
            rebased = _DATA | ((true ^ bit) << 10)
            conditioned = np.empty(2048)
            conditioned[rebased] = weights / total
            posterior = post.apply(conditioned)
            evidence += peak + float(np.log(total))
        result = self._decision(posterior, observed, evidence, False)
        self._records.append(observed)
        return result

    def finish(self, syndrome: Iterable[int]) -> CircuitFrameUpdate:
        """Condition on an actual ideal final reset/extraction, with no new faults."""
        if self._finished:
            raise ValueError("history is already finished")
        observed = _syndrome_bits(syndrome)
        marginal = self.data_posterior
        marginal[(np.arange(1024) & 255) != _index(observed)] = 0
        total = float(marginal.sum())
        if total == 0:
            raise ValueError("zero-probability final circuit syndrome")
        posterior = np.zeros(2048)
        posterior[:1024] = marginal / total
        selected = None
        if self.certification_rounds == self.n_rounds:
            table = certified_classes(self.n_rounds, self.schedule)
            if table is not None:
                initial = _class_index(self._initial_frame)
                offset = tuple((initial >> bit) & 1 for bit in range(8))
                signature = tuple(
                    tuple(bit ^ known for bit, known in zip(bits, offset, strict=True))
                    for bits in (*self._records, observed)
                )
                relative = table.get(signature)
                if relative is not None and marginal[initial ^ relative] > 0:
                    selected = initial ^ relative
        return self._decision(
            posterior, observed, float(np.log(total)), True, selected, selected is not None
        )

    def decode(self, history, initial_frame=None, final_syndrome=None):
        decoder = CircuitHistoryDecoderD3(
            self.noise, self.schedule, initial_frame, certification_rounds=self.certification_rounds
        )
        try:
            observations = iter(history)
        except TypeError:
            raise ValueError("history must contain observed syndrome records") from None
        result = None
        for observed in observations:
            result = decoder.update(observed)
        if result is None:
            raise ValueError("history must contain at least one circuit round")
        return result if final_syndrome is None else decoder.finish(final_syndrome)


@dataclass(frozen=True)
class CircuitCorrectionResult:
    memory: CircuitMemoryResult
    updates: tuple[HistoryFrameUpdate, ...]
    final_update: HistoryFrameUpdate | None = None

    @property
    def state(self):
        return self.memory.state

    @property
    def update(self):
        return self.updates[-1] if self.final_update is None else self.final_update

    @property
    def frame(self):
        return self.update.frame

    @property
    def corrected_state(self):
        return self.frame.apply(self.state)


class CircuitFrameCorrectionD3:
    def __init__(self, noise=None, schedule=None, *, certification_rounds=3):
        self._extractor = CircuitExtractorD3(noise, schedule)
        self.certification_rounds = certification_rounds

    def run(
        self,
        state: Statevector,
        n_rounds: int,
        rng=None,
        *,
        initial_frame=None,
        injections=(),
        final_readout=False,
        sample_noise=True,
    ):
        if initial_frame is not None:
            _check_frame(initial_frame)
        decoder = CircuitHistoryDecoderD3(
            self._extractor.noise,
            self._extractor.schedule,
            initial_frame,
            certification_rounds=self.certification_rounds,
        )
        memory = self._extractor.run(
            state,
            n_rounds,
            rng,
            injections=injections,
            final_readout=final_readout,
            sample_noise=sample_noise,
        )
        updates = tuple(decoder.update(record.syndrome) for record in memory.rounds)
        final = (
            None
            if memory.final_measurement is None
            else decoder.finish(memory.final_measurement.syndrome)
        )
        return CircuitCorrectionResult(memory, updates, final)


# Single-fault audits


@dataclass(frozen=True)
class AuditCase:
    fault: CircuitFault | None
    input_error: PauliString | None
    trace: FaultTrace


@lru_cache(maxsize=16)
def single_fault_catalog(
    n_rounds: int = 3,
    schedule: CircuitScheduleD3 = DEFAULT_SCHEDULE,
) -> tuple[AuditCase, ...]:
    """Clean encoded input; one allowed fault, or one input error with no internal fault.

    Every signature includes an ideal final syndrome extraction with an ideal
    initial ancilla reset and no new faults. Initial encoding itself is ideal.
    """
    n_rounds = _integer(n_rounds, "n_rounds")
    if n_rounds < 1:
        raise ValueError("n_rounds must be positive")
    cases = [AuditCase(None, None, trace_faults((), n_rounds, schedule))]
    for q in range(9):
        for axis in "XYZ":
            error = PauliString("I" * q + axis + "I" * (8 - q))
            cases.append(
                AuditCase(
                    None, error, trace_faults((), n_rounds, schedule, PauliFrame.from_pauli(error))
                )
            )
    for stage in range(-1, n_rounds):
        locations = fault_locations(schedule, preparation=stage == -1)
        for location in locations:
            for alternative in location.alternatives:
                fault = CircuitFault(stage, location.name, alternative)
                cases.append(AuditCase(fault, None, trace_faults((fault,), n_rounds, schedule)))
    return tuple(cases)


@dataclass(frozen=True)
class FaultAudit:
    n_rounds: int
    cases: tuple[AuditCase, ...]
    signatures: int
    ambiguous: tuple[tuple[int, ...], ...]

    @property
    def passed(self) -> bool:
        return not self.ambiguous

    @property
    def internal_faults(self) -> int:
        return sum(case.fault is not None for case in self.cases)


def audit_schedule(
    n_rounds: int = 3,
    schedule: CircuitScheduleD3 = DEFAULT_SCHEDULE,
) -> FaultAudit:
    """Find identical complete records requiring different logical recoveries."""
    cases = single_fault_catalog(n_rounds, schedule)
    groups = defaultdict(list)
    for index, case in enumerate(cases):
        groups[case.trace.signature].append(index)
    ambiguous = tuple(
        tuple(indices)
        for indices in groups.values()
        if len({_class_index(cases[i].trace.final_error) for i in indices}) > 1
    )
    return FaultAudit(n_rounds, cases, len(groups), ambiguous)


class SingleFaultHistoryDecoderD3:
    """Certified lookup for the declared <=1-fault closed-memory experiment only.

    Unknown histories are rejected. This is an audit/recovery reference, not a
    probabilistic decoder for multiple faults. Inputs are records, not fault IDs.
    """

    def __init__(self, n_rounds: int = 3, schedule: CircuitScheduleD3 = DEFAULT_SCHEDULE):
        audit = audit_schedule(n_rounds, schedule)
        if not audit.passed:
            raise ValueError(
                "schedule has indistinguishable single faults with different logical classes"
            )
        frames = _representatives()
        self.n_rounds = audit.n_rounds
        self._table = {
            case.trace.signature: frames[_class_index(case.trace.final_error)]
            for case in audit.cases
        }

    def decode(self, history: Iterable[Iterable[int]], final_syndrome: Iterable[int]) -> PauliFrame:
        try:
            records = tuple(_syndrome_bits(bits) for bits in history)
        except TypeError:
            raise ValueError("history must contain syndrome records") from None
        if len(records) != self.n_rounds:
            raise ValueError("history length does not match the certified memory experiment")
        signature = (*records, _syndrome_bits(final_syndrome))
        if signature not in self._table:
            raise ValueError("history is outside the certified at-most-one-fault catalog")
        return self._table[signature]


@dataclass(frozen=True)
class DecoderFaultAudit:
    audit: FaultAudit
    recoveries: tuple[PauliFrame, ...]
    failures: tuple[int, ...]
    noise: CircuitNoise
    certification: bool

    @property
    def passed(self) -> bool:
        return self.audit.passed and not self.failures

    def to_dict(self) -> dict:
        """Full machine-readable catalogue, including every propagated fault and decision."""
        from dataclasses import asdict

        return {
            "boundaries": (
                "ideal encoded input; noisy ancilla preparation and memory rounds; "
                "ideal final reset/extraction"
            ),
            "noise_prior": asdict(self.noise),
            "certified_decisions": self.certification,
            "n_rounds": self.audit.n_rounds,
            "cases": len(self.audit.cases),
            "internal_fault_cases": self.audit.internal_faults,
            "unique_signatures": self.audit.signatures,
            "ambiguous_groups": self.audit.ambiguous,
            "decoder_failures": self.failures,
            "passed": self.passed,
            "catalog": [
                {
                    "index": index,
                    "fault": None if case.fault is None else asdict(case.fault),
                    "input_error": None if case.input_error is None else case.input_error.labels,
                    "observed_history": case.trace.history,
                    "true_history": case.trace.true_history,
                    "propagated_data_errors": [
                        frame.pauli.labels for frame in case.trace.data_errors
                    ],
                    "ancilla_bits": case.trace.ancilla_bits,
                    "final_syndrome": case.trace.final_syndrome,
                    "recovery": recovery.pauli.labels,
                    "passed": index not in self.failures,
                }
                for index, (case, recovery) in enumerate(
                    zip(self.audit.cases, self.recoveries, strict=True)
                )
            ],
        }


def audit_circuit_decoder(
    n_rounds: int = 3,
    schedule: CircuitScheduleD3 = DEFAULT_SCHEDULE,
    noise=None,
    *,
    certify: bool = True,
) -> DecoderFaultAudit:
    """Test the deployed all-order Bayesian decoder on every <=1-fault case.

    Rates specify the decoder's prior, not extra sampled faults during this
    deterministic test. The default is a low-rate uniform circuit model.
    """

    audit = audit_schedule(n_rounds, schedule)
    model = CircuitNoise.uniform(1e-5) if noise is None else noise
    decoder = CircuitHistoryDecoderD3(
        model, schedule, certification_rounds=n_rounds if certify else None
    )
    cache = {}
    recoveries, failures = [], []
    for index, case in enumerate(audit.cases):
        signature = case.trace.signature
        if signature not in cache:
            cache[signature] = decoder.decode(
                case.trace.history, final_syndrome=case.trace.final_syndrome
            ).frame
        recovery = cache[signature]
        recoveries.append(recovery)
        residual = case.trace.final_error.updated(recovery.pauli)
        if _class_index(residual) != 0:
            failures.append(index)
    return DecoderFaultAudit(audit, tuple(recoveries), tuple(failures), model, certify)


@lru_cache(maxsize=16)
def certified_classes(n_rounds: int, schedule: CircuitScheduleD3) -> dict | None:
    """Cached certificate lookup; ambiguity forbids a certificate."""
    audit = audit_schedule(n_rounds, schedule)
    if not audit.passed:
        return None
    return {case.trace.signature: _class_index(case.trace.final_error) for case in audit.cases}


# Memory benchmarks


def wilson_interval(failures: int, trials: int) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval; zero observed failures is not zero risk."""
    failures, trials = _integer(failures, "failures"), _integer(trials, "trials")
    if trials < 1 or not 0 <= failures <= trials:
        raise ValueError("require trials > 0 and 0 <= failures <= trials")
    z = 1.959963984540054
    frequency = failures / trials
    denominator = 1 + z * z / trials
    center = (frequency + z * z / (2 * trials)) / denominator
    radius = (
        z * sqrt(frequency * (1 - frequency) / trials + z * z / (4 * trials * trials)) / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def two_fault_probability(noise: CircuitNoise, n_rounds: int, schedule: CircuitScheduleD3) -> float:
    """Exact P(at least two independent location events), including zero-effect Paulis."""
    n_rounds = _integer(n_rounds, "n_rounds")
    if n_rounds < 1:
        raise ValueError("n_rounds must be positive")
    locations = (
        *fault_locations(schedule, preparation=True),
        *(
            location
            for _ in range(n_rounds)
            for operation in circuit_operations(schedule)
            for location in operation.locations
        ),
    )
    zero, one = 1.0, 0.0
    for location in locations:
        p = getattr(noise, location.rate)
        zero, one = zero * (1 - p), one * (1 - p) + zero * p
    return max(0.0, min(1.0, 1 - zero - one))


def benchmark_circuit_memory(
    probability: float,
    trials: int = 1000,
    n_rounds: int = 3,
    seed: int = 42,
    *,
    final_readout: bool = True,
) -> dict:
    """Uniform-rate benchmark; all singleton cases must pass before shots are sampled.

    The no-noise control uses the positive-rate structural audit; no fault is
    physically allowed in that control. Nonzero runs audit their actual prior.
    The single-fault guarantee applies to the declared closed experiment only.
    """
    trials, n_rounds, seed = (
        _integer(value, name)
        for value, name in ((trials, "trials"), (n_rounds, "n_rounds"), (seed, "seed"))
    )
    if trials < 1 or n_rounds < 1 or seed < 0:
        raise ValueError("trials/rounds must be positive and seed nonnegative")
    if not isinstance(final_readout, bool):
        raise ValueError("final_readout must be Boolean")
    noise = CircuitNoise.uniform(probability)
    schedule = CircuitScheduleD3()
    audit_model = noise if probability > 0 else CircuitNoise.uniform(1e-5)
    audit = audit_circuit_decoder(n_rounds, schedule, audit_model)
    if not audit.passed:
        raise RuntimeError("single-fault audit failed; statistical benchmark refused")
    code = RotatedSurfaceCodeD3()
    initial = code.encode(np.sqrt(0.37), np.exp(0.43j) * np.sqrt(0.63))
    runner = CircuitFrameCorrectionD3(noise, schedule, certification_rounds=n_rounds)
    generator = np.random.default_rng(seed)
    failures = raw_failures = singleton_shots = singleton_failures = certificate_decisions = 0
    fault_counts = {"zero": 0, "one": 0, "multiple": 0}
    for _ in range(trials):
        result = runner.run(initial, n_rounds, generator, final_readout=final_readout)
        faults = (
            *result.memory.preparation_faults,
            *(fault for record in result.memory.rounds for fault in record.faults),
        )
        trace = trace_faults(faults, n_rounds, schedule)
        if tuple(record.syndrome for record in result.memory.rounds) != trace.history:
            raise RuntimeError("dense circuit and independent symbolic record disagree")
        if final_readout and result.memory.final_measurement.syndrome != trace.final_syndrome:
            raise RuntimeError("physical clean closure disagrees with independent fault trace")
        residual = trace.final_error.updated(result.frame.pauli)
        failed = _class_index(residual) != 0
        actual_fidelity = abs(np.vdot(initial.amplitudes, result.corrected_state.amplitudes)) ** 2
        if failed == bool(np.isclose(actual_fidelity, 1, atol=1e-11, rtol=0)):
            raise RuntimeError("stabilizer-class and statevector recovery tests disagree")
        failures += failed
        raw_failures += _class_index(trace.final_error) != 0
        certificate_decisions += result.update.single_fault_certified
        bucket = "zero" if not faults else "one" if len(faults) == 1 else "multiple"
        fault_counts[bucket] += 1
        if len(faults) == 1:
            singleton_shots += 1
            singleton_failures += failed
        if final_readout and len(faults) <= 1 and failed:
            raise RuntimeError("sampled singleton contradicted its closed-memory certificate")
    interval = wilson_interval(failures, trials)
    return {
        "noise": asdict(noise),
        "seed": seed,
        "trials": trials,
        "n_rounds": n_rounds,
        "final_readout": final_readout,
        "decision_policy": "certified closed histories; Bayesian MAP otherwise",
        "audit_prior": asdict(audit_model),
        "audit_cases": len(audit.audit.cases),
        "audit_failures": len(audit.failures),
        "fault_count_shots": fault_counts,
        "singleton_shots": singleton_shots,
        "singleton_failures": singleton_failures,
        "certificate_decisions": certificate_decisions,
        "raw_failures": raw_failures,
        "failures": failures,
        "failure_rate": failures / trials,
        "failure_rate_wilson95": interval,
        "two_or_more_fault_probability": two_fault_probability(noise, n_rounds, schedule),
        "bound_applicability": "closed-memory protocol only; open tails are not certified",
    }
