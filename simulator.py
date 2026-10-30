

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Statevectors and Pauli operators

ComplexVector = NDArray[np.complex128]
_ATOL = 1e-12
_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class PauliString:
    """Hermitian tensor product with a real sign (+1 or -1).

    ``PauliString('XZ')`` means X on q0 and Z on q1, i.e. Z tensor X
    in conventional matrix order. No implicit string reversal is performed.
    """

    labels: str
    sign: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.labels, str) or not self.labels:
            raise ValueError("labels must be a nonempty I/X/Y/Z string")
        if any(label not in "IXYZ" for label in self.labels):
            raise ValueError("labels may contain only I, X, Y, Z")
        if isinstance(self.sign, bool) or self.sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")

    @property
    def n_qubits(self) -> int:
        return len(self.labels)

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(q for q, label in enumerate(self.labels) if label != "I")

    @property
    def weight(self) -> int:
        return len(self.support)

    def commutes_with(self, other: "PauliString") -> bool:
        if self.n_qubits != other.n_qubits:
            raise ValueError("Pauli lengths must agree")
        conflicts = sum(
            a != b and a != "I" and b != "I" for a, b in zip(self.labels, other.labels, strict=True)
        )
        return conflicts % 2 == 0

    def apply_to(self, amplitudes: ArrayLike) -> ComplexVector:
        """Return P|psi> without constructing a 2**n by 2**n matrix."""
        vector = np.asarray(amplitudes, dtype=np.complex128)
        if vector.shape != (1 << self.n_qubits,):
            raise ValueError("amplitude shape does not match the Pauli length")
        indices = np.arange(vector.size, dtype=np.int64)
        destinations = indices.copy()
        phases = np.full(vector.size, self.sign, dtype=np.complex128)
        for q, label in enumerate(self.labels):
            bit_is_one = (indices & (1 << q)) != 0
            if label in "XY":
                destinations ^= 1 << q
            if label in "YZ":
                phases *= np.where(bit_is_one, -1, 1)
            if label == "Y":
                phases *= 1j
        result = np.empty_like(vector)
        result[destinations] = phases * vector
        return result


class Statevector:
    """Normalized pure state stored in a dense, owned complex128 array.

    Construction rejects non-normalized inputs rather than silently changing
    them. ``amplitudes`` returns a copy so callers cannot corrupt invariants.
    Memory for amplitudes alone is 16 * 2**n bytes; operations need temporaries.
    """

    def __init__(self, amplitudes: ArrayLike):
        vector = np.array(amplitudes, dtype=np.complex128, copy=True)
        if vector.ndim != 1 or vector.size < 2 or vector.size & (vector.size - 1):
            raise ValueError("amplitudes must be a 1D array of length 2**n, n >= 1")
        norm_squared = float(np.vdot(vector, vector).real)
        if not np.all(np.isfinite(vector)) or not np.isclose(norm_squared, 1.0, atol=_ATOL, rtol=0):
            raise ValueError("amplitudes must be finite and normalized")
        self._amplitudes = vector
        self.n_qubits = vector.size.bit_length() - 1

    @classmethod
    def basis(cls, n_qubits: int, index: int = 0) -> "Statevector":
        n_qubits = _integer(n_qubits, "n_qubits")
        index = _integer(index, "index")
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if not 0 <= index < (1 << n_qubits):
            raise ValueError("basis index out of range")
        amplitudes = np.zeros(1 << n_qubits, dtype=np.complex128)
        amplitudes[index] = 1
        return cls(amplitudes)

    @property
    def amplitudes(self) -> ComplexVector:
        return self._amplitudes.copy()

    def copy(self) -> "Statevector":
        return Statevector(self._amplitudes)

    def _check_qubit(self, q: int) -> int:
        q = _integer(q, "qubit")
        if not 0 <= q < self.n_qubits:
            raise ValueError("qubit index out of range")
        return q

    def _check_pauli(self, pauli: PauliString) -> None:
        if pauli.n_qubits != self.n_qubits:
            raise ValueError("Pauli length does not match state size")

    def apply_gate(self, gate: ArrayLike, q: int) -> "Statevector":
        """Apply a validated 2x2 unitary and correct numerical norm drift."""
        q = self._check_qubit(q)
        matrix = np.asarray(gate, dtype=np.complex128)
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise ValueError("gate must be a finite 2x2 matrix")
        if not np.allclose(matrix.conj().T @ matrix, np.eye(2), atol=_ATOL, rtol=0):
            raise ValueError("gate must be unitary")
        indices = np.arange(self._amplitudes.size)
        low = indices[(indices & (1 << q)) == 0]
        high = low | (1 << q)
        a, b = self._amplitudes[low], self._amplitudes[high]
        self._amplitudes[low] = matrix[0, 0] * a + matrix[0, 1] * b
        self._amplitudes[high] = matrix[1, 0] * a + matrix[1, 1] * b
        # Accepted unitarity error and roundoff must not accumulate across gates.
        self._amplitudes /= np.linalg.norm(self._amplitudes)
        return self

    def h(self, q: int) -> "Statevector":
        return self.apply_gate(_H, q)

    def x(self, q: int) -> "Statevector":
        return self.apply_gate([[0, 1], [1, 0]], q)

    def y(self, q: int) -> "Statevector":
        return self.apply_gate([[0, -1j], [1j, 0]], q)

    def z(self, q: int) -> "Statevector":
        return self.apply_gate([[1, 0], [0, -1]], q)

    def s(self, q: int) -> "Statevector":
        return self.apply_gate([[1, 0], [0, 1j]], q)

    def sdg(self, q: int) -> "Statevector":
        return self.apply_gate([[1, 0], [0, -1j]], q)

    def t(self, q: int) -> "Statevector":
        return self.apply_gate([[1, 0], [0, np.exp(1j * np.pi / 4)]], q)

    def tdg(self, q: int) -> "Statevector":
        return self.apply_gate([[1, 0], [0, np.exp(-1j * np.pi / 4)]], q)

    def _pair(self, a: int, b: int) -> tuple[int, int]:
        a, b = self._check_qubit(a), self._check_qubit(b)
        if a == b:
            raise ValueError("two-qubit gates require distinct qubits")
        return a, b

    def cnot(self, control: int, target: int) -> "Statevector":
        control, target = self._pair(control, target)
        indices = np.arange(self._amplitudes.size)
        destinations = indices ^ (((indices >> control) & 1) << target)
        self._amplitudes = self._amplitudes[destinations]
        return self

    def ccx(self, control_a: int, control_b: int, target: int) -> "Statevector":
        """Ideal Toffoli: flip target iff both distinct controls are one.

        This primitive acts coherently on arbitrary states. It is not a
        fault-tolerant encoded gate or a physical Clifford+T decomposition.
        """
        control_a, control_b = self._pair(control_a, control_b)
        target = self._check_qubit(target)
        if target in (control_a, control_b):
            raise ValueError("Toffoli requires three distinct qubits")
        indices = np.arange(self._amplitudes.size)
        enabled = ((indices >> control_a) & (indices >> control_b)) & 1
        destinations = indices ^ (enabled << target)
        self._amplitudes = self._amplitudes[destinations]
        return self

    def cz(self, a: int, b: int) -> "Statevector":
        a, b = self._pair(a, b)
        indices = np.arange(self._amplitudes.size)
        self._amplitudes[((indices >> a) & (indices >> b) & 1) == 1] *= -1
        return self

    def swap(self, a: int, b: int) -> "Statevector":
        a, b = self._pair(a, b)
        indices = np.arange(self._amplitudes.size)
        different = ((indices >> a) ^ (indices >> b)) & 1
        destinations = indices ^ (different << a) ^ (different << b)
        self._amplitudes = self._amplitudes[destinations]
        return self

    def apply_pauli(self, pauli: PauliString) -> "Statevector":
        self._check_pauli(pauli)
        self._amplitudes = pauli.apply_to(self._amplitudes)
        return self

    def expectation(self, pauli: PauliString) -> float:
        self._check_pauli(pauli)
        return float(np.vdot(self._amplitudes, pauli.apply_to(self._amplitudes)).real)

    def probabilities(self) -> NDArray[np.float64]:
        return np.abs(self._amplitudes) ** 2

    def _projected_amplitudes(self, pauli: PauliString, outcome: int) -> ComplexVector:
        self._check_pauli(pauli)
        if isinstance(outcome, bool) or outcome not in (-1, 1):
            raise ValueError("outcome must be +1 or -1")
        return (self._amplitudes + outcome * pauli.apply_to(self._amplitudes)) / 2

    def probability(self, pauli: PauliString, outcome: int = 1) -> float:
        """Born probability of an ideal Pauli measurement eigenvalue +/-1."""
        projected = self._projected_amplitudes(pauli, outcome)
        return float(np.clip(np.vdot(projected, projected).real, 0, 1))

    def postselect(self, pauli: PauliString, outcome: int = 1) -> float:
        """Apply (I + outcome*P)/2, normalize, and return its probability.

        This is an ideal mathematical projector, not an ancilla circuit.
        Impossible (numerically zero) outcomes raise without changing the state.
        """
        projected = self._projected_amplitudes(pauli, outcome)
        norm_squared = float(np.vdot(projected, projected).real)
        if norm_squared <= np.finfo(np.float64).eps ** 2:
            raise ValueError("cannot postselect a zero-probability outcome")
        self._amplitudes = projected / np.sqrt(norm_squared)
        return float(np.clip(norm_squared, 0, 1))

    def measure_pauli(self, pauli: PauliString, rng: np.random.Generator | None = None) -> int:
        """Sample +/-1 and collapse; pass a seeded Generator for reproducibility."""
        probability_plus = self.probability(pauli)
        generator = np.random.default_rng() if rng is None else rng
        # Handle deterministic cases without a draw or roundoff-induced rare branch.
        if probability_plus <= np.finfo(np.float64).eps ** 2:
            outcome = -1
        elif self.probability(pauli, -1) <= np.finfo(np.float64).eps ** 2:
            outcome = 1
        else:
            outcome = 1 if generator.random() < probability_plus else -1
        self.postselect(pauli, outcome)
        return outcome

    def measure_z(self, q: int, rng: np.random.Generator | None = None) -> int:
        """Sample a computational-basis bit (0 or 1) and collapse qubit q."""
        q = self._check_qubit(q)
        labels = "I" * q + "Z" + "I" * (self.n_qubits - q - 1)
        return (1 - self.measure_pauli(PauliString(labels), rng)) // 2


# Correction frames


@dataclass(frozen=True)
class PauliFrame:
    """Correction convention: corrected_state = F * raw_state, up to global phase.

    Bit q of x_mask/z_mask records the X/Z component on qubit q; both bits
    denote Y. Updates and propagation return new frames without changing data.
    """

    n_qubits: int
    x_mask: int = 0
    z_mask: int = 0

    def __post_init__(self) -> None:
        n = _integer(self.n_qubits, "n_qubits")
        if n < 1:
            raise ValueError("n_qubits must be positive")
        object.__setattr__(self, "n_qubits", n)
        for name in ("x_mask", "z_mask"):
            mask = _integer(getattr(self, name), name)
            if not 0 <= mask < (1 << n):
                raise ValueError(f"{name} must fit in n_qubits bits")
            object.__setattr__(self, name, mask)

    @classmethod
    def from_pauli(cls, pauli: PauliString) -> "PauliFrame":
        """Represent a Pauli, ignoring its physically irrelevant overall sign."""
        x = sum((label in "XY") << q for q, label in enumerate(pauli.labels))
        z = sum((label in "YZ") << q for q, label in enumerate(pauli.labels))
        return cls(pauli.n_qubits, x, z)

    @property
    def pauli(self) -> PauliString:
        """Canonical Hermitian representative of this frame, with sign +1."""
        return PauliString(
            "".join(
                "IXZY"[((self.x_mask >> q) & 1) | (((self.z_mask >> q) & 1) << 1)]
                for q in range(self.n_qubits)
            )
        )

    def updated(self, recovery: PauliString) -> "PauliFrame":
        """Return recovery * F, modulo global phase; do not mutate the frame."""
        other = PauliFrame.from_pauli(recovery)
        if other.n_qubits != self.n_qubits:
            raise ValueError("recovery length must match frame size")
        return PauliFrame(self.n_qubits, self.x_mask ^ other.x_mask, self.z_mask ^ other.z_mask)

    def apply(self, state: Statevector) -> Statevector:
        """Materialize the correction on an owned copy, leaving frame/data unchanged.

        If continuing with this corrected copy, use an identity frame to avoid
        applying the pending correction a second time.
        """
        if state.n_qubits != self.n_qubits:
            raise ValueError("state size must match frame size")
        return state.copy().apply_pauli(self.pauli)

    def sign(self, observable: PauliString) -> int:
        """Sign in F† observable F = sign * observable."""
        other = PauliFrame.from_pauli(observable)
        if other.n_qubits != self.n_qubits:
            raise ValueError("observable length must match frame size")
        parity = (
            (self.x_mask & other.z_mask).bit_count() + (self.z_mask & other.x_mask).bit_count()
        ) % 2
        return 1 - 2 * parity

    def adjust_outcome(self, observable: PauliString, outcome: int) -> int:
        """Translate a raw +/-1 Pauli eigenvalue into the corrected frame."""
        outcome = _integer(outcome, "outcome")
        if outcome not in (-1, 1):
            raise ValueError("outcome must be +1 or -1")
        return self.sign(observable) * outcome

    def expectation(self, state: Statevector, observable: PauliString) -> float:
        """Correct a raw expectation without applying correction gates to data."""
        if state.n_qubits != self.n_qubits:
            raise ValueError("state size must match frame size")
        return self.sign(observable) * state.expectation(observable)

    def propagated(self, gate: str, *qubits: int) -> "PauliFrame":
        """Return U F U† for a supported Clifford, ignoring global phase.

        The caller must apply the same gate U to the raw state. Supported names
        match Statevector: h, s, sdg, x, y, z, cnot, cz, swap. Non-Clifford gates
        require an explicit separate treatment; T/TDG are rejected.
        """
        single = ("h", "s", "sdg", "x", "y", "z")
        pair = ("cnot", "cz", "swap")
        if gate not in single + pair:
            raise ValueError("unsupported gate for a Pauli frame; use a named Clifford gate")
        if len(qubits) != (1 if gate in single else 2):
            raise ValueError("wrong number of qubits for Clifford gate")
        targets = tuple(_integer(q, "qubit") for q in qubits)
        if any(not 0 <= q < self.n_qubits for q in targets):
            raise ValueError("qubit index out of range")
        if len(targets) == 2 and targets[0] == targets[1]:
            raise ValueError("two-qubit gates require distinct qubits")
        x, z = self.x_mask, self.z_mask
        a = targets[0]
        if gate == "h":
            difference = ((x >> a) ^ (z >> a)) & 1
            x ^= difference << a
            z ^= difference << a
        elif gate in ("s", "sdg"):
            z ^= ((x >> a) & 1) << a
        elif gate == "cnot":
            b = targets[1]
            x ^= ((x >> a) & 1) << b
            z ^= ((z >> b) & 1) << a
        elif gate == "cz":
            b = targets[1]
            z ^= (((x >> a) & 1) << b) | (((x >> b) & 1) << a)
        elif gate == "swap":
            b = targets[1]
            x_difference = ((x >> a) ^ (x >> b)) & 1
            z_difference = ((z >> a) ^ (z >> b)) & 1
            x ^= (x_difference << a) | (x_difference << b)
            z ^= (z_difference << a) | (z_difference << b)
        return PauliFrame(self.n_qubits, x, z)
