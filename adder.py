"""Reversible logical addition using CNOT and Toffoli gates."""

from dataclasses import dataclass

import numpy as np

from simulator import Statevector, _integer

# Adder circuit


@dataclass(frozen=True)
class LogicalGate:
    """An ideal CNOT or Toffoli on logical wire indices; target is last."""

    gate: str
    targets: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.gate, str) or self.gate not in ("cnot", "ccx"):
            raise ValueError("logical adder gates must be cnot or ccx")
        try:
            targets = tuple(_integer(q, "logical wire") for q in self.targets)
        except TypeError:
            raise ValueError("targets must contain logical wire indices") from None
        arity = 2 if self.gate == "cnot" else 3
        if len(targets) != arity or len(set(targets)) != arity or min(targets) < 0:
            raise ValueError("gate targets must be distinct nonnegative wires of the correct arity")
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True)
class AdderLayout:
    """Little-endian registers A[n], B[n], SUM[n+1], then two workspace wires."""

    n_bits: int

    def __post_init__(self):
        n = _integer(self.n_bits, "n_bits")
        if n < 1:
            raise ValueError("n_bits must be positive")
        object.__setattr__(self, "n_bits", n)

    @property
    def a(self) -> tuple[int, ...]:
        return tuple(range(self.n_bits))

    @property
    def b(self) -> tuple[int, ...]:
        return tuple(range(self.n_bits, 2 * self.n_bits))

    @property
    def sum(self) -> tuple[int, ...]:
        return tuple(range(2 * self.n_bits, 3 * self.n_bits + 1))

    @property
    def workspace(self) -> tuple[int, int]:
        return 3 * self.n_bits + 1, 3 * self.n_bits + 2

    @property
    def n_qubits(self) -> int:
        return 3 * self.n_bits + 3

    @property
    def names(self) -> tuple[str, ...]:
        return (
            *(f"a{i}" for i in range(self.n_bits)),
            *(f"b{i}" for i in range(self.n_bits)),
            *(f"sum{i}" for i in range(self.n_bits + 1)),
            "work_cin",
            "work_cout",
        )


@dataclass(frozen=True)
class AdditionSample:
    """One computational-basis result; total includes the high carry bit."""

    a: int
    b: int
    total: int
    workspace: tuple[int, int]


def _ripple(layout):
    """In-place full adder: restore A/cin, write sum in B and xor carry into cout."""
    gates = []
    for i, (a, b) in enumerate(zip(layout.a, layout.b, strict=True)):
        carry = layout.workspace[0] if i == 0 else layout.a[i - 1]
        # MAJ(a,b,c): carry out is held in a.
        gates.extend(
            (
                LogicalGate("cnot", (a, b)),
                LogicalGate("cnot", (a, carry)),
                LogicalGate("ccx", (b, carry, a)),
            )
        )
    gates.append(LogicalGate("cnot", (layout.a[-1], layout.workspace[1])))
    for i in reversed(range(layout.n_bits)):
        a, b = layout.a[i], layout.b[i]
        carry = layout.workspace[0] if i == 0 else layout.a[i - 1]
        # UMA restores a/c and writes a XOR b XOR c into b.
        gates.extend(
            (
                LogicalGate("ccx", (b, carry, a)),
                LogicalGate("cnot", (a, carry)),
                LogicalGate("cnot", (carry, b)),
            )
        )
    return tuple(gates)


class ReversibleAdder:
    """Ideal unsigned n-bit oracle: |a,b,s,00> -> |a,b,s XOR (a+b),00>.

    A separate n+1-bit output prevents arithmetic overflow. On a zero output
    register this is ordinary addition. The input registers are preserved.
    This logical reference supplies neither surface-code execution nor noise.
    """

    def __init__(self, n_bits: int):
        self._layout = AdderLayout(n_bits)
        core = _ripple(self.layout)
        copy_sum = tuple(
            LogicalGate("cnot", (b, out))
            for b, out in zip(self.layout.b, self.layout.sum[:-1], strict=True)
        ) + (LogicalGate("cnot", (self.layout.workspace[1], self.layout.sum[-1])),)
        self._gates = (*core, *copy_sum, *reversed(core))

    @property
    def layout(self) -> AdderLayout:
        return self._layout

    @property
    def n_bits(self) -> int:
        return self.layout.n_bits

    @property
    def n_qubits(self) -> int:
        return self.layout.n_qubits

    @property
    def gates(self) -> tuple[LogicalGate, ...]:
        return self._gates

    @property
    def gate_counts(self) -> dict[str, int]:
        return {gate: sum(op.gate == gate for op in self.gates) for gate in ("cnot", "ccx")}

    @property
    def statevector_bytes(self) -> int:
        """Amplitude storage alone; executing gates requires additional arrays."""
        return 16 * (1 << self.n_qubits)

    def _budget(self, n_qubits, max_qubits):
        maximum = _integer(max_qubits, "max_qubits")
        if maximum < 1:
            raise ValueError("max_qubits must be positive")
        if n_qubits > maximum:
            raise ValueError(
                f"dense simulation needs {n_qubits} logical qubits, exceeding "
                f"max_qubits={maximum}; use simulate_basis for definite binary inputs, "
                "or explicitly increase the limit if memory permits"
            )

    def _mapping(self, qubits, width=None):
        if qubits is None:
            mapped = tuple(range(self.n_qubits))
        else:
            try:
                mapped = tuple(_integer(q, "logical wire") for q in qubits)
            except TypeError:
                raise ValueError("qubits must be a complete logical-wire mapping") from None
        if (
            len(mapped) != self.n_qubits
            or len(set(mapped)) != self.n_qubits
            or min(mapped) < 0
            or (width is not None and max(mapped) >= width)
        ):
            raise ValueError(
                "wire mapping must be complete, distinct, nonnegative, and fit the state"
            )
        return mapped

    def operations(self, qubits=None, *, inverse: bool = False) -> tuple[LogicalGate, ...]:
        """Export/relabel logical gates for later circuit composition."""
        if not isinstance(inverse, bool):
            raise ValueError("inverse must be Boolean")
        mapping = self._mapping(qubits)
        gates = reversed(self.gates) if inverse else self.gates
        return tuple(LogicalGate(op.gate, tuple(mapping[q] for q in op.targets)) for op in gates)

    def _input_index(self, a, b, initial_sum):
        a, b, initial_sum = (
            _integer(value, name)
            for value, name in ((a, "a"), (b, "b"), (initial_sum, "initial_sum"))
        )
        if not 0 <= a < (1 << self.n_bits) or not 0 <= b < (1 << self.n_bits):
            raise ValueError("inputs must be unsigned integers fitting n_bits")
        if not 0 <= initial_sum < (1 << (self.n_bits + 1)):
            raise ValueError("initial_sum must fit the n_bits+1 output register")
        return a | (b << self.n_bits) | (initial_sum << (2 * self.n_bits))

    def basis_input(self, a: int, b: int, *, initial_sum=0, max_qubits=24) -> Statevector:
        """Allocate a dense logical basis input with both workspace wires at zero."""
        self._budget(self.n_qubits, max_qubits)
        return Statevector.basis(self.n_qubits, self._input_index(a, b, initial_sum))

    def run(self, state: Statevector, *, qubits=None, inverse=False, max_qubits=24) -> Statevector:
        """Apply the gate circuit coherently to an owned copy of a shared state.

        Workspace must start at zero; A, B, and SUM can be arbitrarily entangled
        with one another or spectator wires. An explicit mapping permits use
        within a larger logical state. Extra unmapped wires are untouched.
        """
        if not isinstance(state, Statevector):
            raise ValueError("state must be Statevector")
        self._budget(state.n_qubits, max_qubits)
        mapping = self._mapping(qubits, state.n_qubits)
        operations = self.operations(mapping, inverse=inverse)
        # Inspect workspace without projecting or discarding any amplitude.
        amplitudes = state.amplitudes
        for local in self.layout.workspace:
            if np.any(amplitudes.reshape(-1, 2, 1 << mapping[local])[:, 1, :] != 0):
                raise ValueError("adder workspace must start at |00>")
        del amplitudes
        current = state.copy()
        for op in operations:
            getattr(current, op.gate)(*op.targets)
        return current

    def _read_index(self, index, mapping):
        def value(wires):
            return sum(((index >> mapping[q]) & 1) << bit for bit, q in enumerate(wires))

        return AdditionSample(
            value(self.layout.a),
            value(self.layout.b),
            value(self.layout.sum),
            tuple((index >> mapping[q]) & 1 for q in self.layout.workspace),
        )

    def simulate_basis(self, a: int, b: int, *, initial_sum=0) -> AdditionSample:
        """Execute the same gates on a definite bitstring, without a statevector.

        This mode has no coherent superpositions. Integer addition is not used
        to produce the result: each CNOT/Toffoli flips its target explicitly.
        """
        index = self._input_index(a, b, initial_sum)
        for op in self.gates:
            if all((index >> q) & 1 for q in op.targets[:-1]):
                index ^= 1 << op.targets[-1]
        return self._read_index(index, tuple(range(self.n_qubits)))

    def sum_probabilities(self, state: Statevector, *, qubits=None) -> np.ndarray:
        """Nonmutating computational-basis marginal over the complete sum register."""
        if not isinstance(state, Statevector):
            raise ValueError("state must be Statevector")
        mapping = self._mapping(qubits, state.n_qubits)
        indices = np.arange(1 << state.n_qubits)
        totals = np.zeros(indices.size, dtype=np.int64)
        for bit, q in enumerate(self.layout.sum):
            totals |= ((indices >> mapping[q]) & 1) << bit
        return np.bincount(totals, weights=state.probabilities(), minlength=1 << (self.n_bits + 1))

    def sample(
        self, state: Statevector, shots=1, rng=None, *, qubits=None
    ) -> tuple[AdditionSample, ...]:
        """Sample independent computational-basis shots without modifying state."""
        shots = _integer(shots, "shots")
        if shots < 1 or not isinstance(state, Statevector):
            raise ValueError("shots must be positive and state must be Statevector")
        mapping = self._mapping(qubits, state.n_qubits)
        generator = np.random.default_rng() if rng is None else rng
        probabilities = state.probabilities()
        probabilities /= probabilities.sum()
        indices = generator.choice(probabilities.size, size=shots, p=probabilities)
        return tuple(self._read_index(int(index), mapping) for index in indices)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Simulate reversible logical addition.")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--a", type=int, default=11)
    parser.add_argument("--b", type=int, default=9)
    parser.add_argument("--mode", choices=("statevector", "basis"), default="statevector")
    parser.add_argument("--max-qubits", type=int, default=24)
    args = parser.parse_args()
    try:
        adder = ReversibleAdder(args.bits)
        if args.mode == "basis":
            result = adder.simulate_basis(args.a, args.b)
        else:
            initial = adder.basis_input(args.a, args.b, max_qubits=args.max_qubits)
            output = adder.run(initial, max_qubits=args.max_qubits)
            (result,) = adder.sample(output)
    except ValueError as error:
        parser.error(str(error))
    print(f"{result.a} + {result.b} = {result.total}")
    print(f"{adder.n_qubits} logical wires; {adder.gate_counts}; workspace {result.workspace}")


if __name__ == "__main__":
    main()
