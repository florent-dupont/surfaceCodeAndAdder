"""Integration checks. Run with: python check.py"""

import unittest
from itertools import permutations

import numpy as np
from numpy.testing import assert_allclose

from adder import ReversibleAdder
from logical_qubits import LogicalRegisterD3
from simulator import PauliString, Statevector
from surface_code import (
    CircuitFault,
    CircuitFrameCorrectionD3,
    CircuitNoise,
    CircuitScheduleD3,
    MinimumWeightDecoderD3,
    RotatedSurfaceCodeD3,
    SyndromeExtractorD3,
    audit_circuit_decoder,
    audit_schedule,
)


def fidelity(left, right):
    return abs(np.vdot(left.amplitudes, right.amplitudes)) ** 2


class Checks(unittest.TestCase):
    def test_toffoli(self):
        rng = np.random.default_rng(31)
        values = rng.normal(size=16) + 1j * rng.normal(size=16)
        values /= np.linalg.norm(values)
        for a, b, target in permutations(range(4), 3):
            expected = np.zeros(16, dtype=complex)
            for index, amplitude in enumerate(values):
                bits = [(index >> q) & 1 for q in range(4)]
                if bits[a] and bits[b]:
                    bits[target] = 1 - bits[target]
                expected[sum(bit << q for q, bit in enumerate(bits))] = amplitude
            actual = Statevector(values).ccx(a, b, target)
            assert_allclose(actual.amplitudes, expected, atol=0, rtol=0)

    def test_coherent_addition(self):
        rng = np.random.default_rng(71)
        for n in range(1, 5):
            adder = ReversibleAdder(n)
            size = 1 << (3 * n + 1)
            values = rng.normal(size=size) + 1j * rng.normal(size=size)
            values /= np.linalg.norm(values)
            source = np.zeros(1 << adder.n_qubits, dtype=complex)
            source[:size] = values
            expected = np.zeros_like(source)
            mask = (1 << n) - 1
            for index, amplitude in enumerate(values):
                a, b, output = index & mask, (index >> n) & mask, index >> (2 * n)
                destination = a | (b << n) | ((output ^ (a + b)) << (2 * n))
                expected[destination] = amplitude
            initial = Statevector(source)
            actual = adder.run(initial)
            assert_allclose(actual.amplitudes, expected, atol=0, rtol=0)
            assert_allclose(adder.run(actual, inverse=True).amplitudes, source, atol=0, rtol=0)
            assert_allclose(initial.amplitudes, source, atol=0, rtol=0)

    def test_binary_addition(self):
        for n in range(1, 7):
            adder = ReversibleAdder(n)
            for a in range(1 << n):
                for b in range(1 << n):
                    result = adder.simulate_basis(a, b)
                    self.assertEqual(
                        (result.a, result.b, result.total, result.workspace), (a, b, a + b, (0, 0))
                    )
        maximum = (1 << 64) - 1
        self.assertEqual(ReversibleAdder(64).simulate_basis(maximum, maximum).total, 2 * maximum)

    def test_single_data_errors(self):
        code = RotatedSurfaceCodeD3()
        initial = code.encode(np.sqrt(0.37), np.exp(0.43j) * np.sqrt(0.63))
        decoder = MinimumWeightDecoderD3()
        extractor = SyndromeExtractorD3()
        for q in range(9):
            for axis in "XYZ":
                error = PauliString("I" * q + axis + "I" * (8 - q))
                measured = extractor.extract(initial.copy().apply_pauli(error))
                corrected = decoder.correct(measured.state, measured.syndrome)
                assert_allclose(fidelity(initial, corrected), 1, atol=1e-12, rtol=0)

    def test_circuit_audit(self):
        report = audit_circuit_decoder()
        self.assertTrue(report.passed)
        self.assertEqual(len(report.audit.cases), 4928)
        legacy = audit_schedule(schedule=CircuitScheduleD3.legacy())
        self.assertEqual(len(legacy.ambiguous), 17)

    def test_continuing_memory(self):
        initial = RotatedSurfaceCodeD3().logical_state("+")
        noise = CircuitNoise.uniform(0.001)
        fault = CircuitFault(0, "Z3.measure.fault", "flip")
        memory = LogicalRegisterD3(["a"])["a"].start_memory(
            initial, noise=noise, sample_noise=False
        )
        memory.step(injections=(fault,), sample_noise=False)
        self.assertEqual(memory.history[0], (0,) * 7 + (1,))
        memory.step(sample_noise=False)
        self.assertEqual(memory.history[1], (1,) + (0,) * 7)
        memory.step(sample_noise=False)
        final = memory.finish()
        self.assertTrue(final.update.single_fault_certified)
        batch = CircuitFrameCorrectionD3(noise).run(
            initial, 3, injections=(fault,), sample_noise=False, final_readout=True
        )
        self.assertEqual(memory.frame, batch.frame)
        assert_allclose(
            fidelity(memory.corrected_state, batch.corrected_state), 1, atol=1e-12, rtol=0
        )

    def test_shared_patches(self):
        register = LogicalRegisterD3(["a", "b"])
        bell = Statevector([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
        shared = register.encode(bell)
        measured = register["b"].measure_ideal_round(shared)
        assert_allclose(
            register.decode(measured.state).amplitudes, bell.amplitudes, atol=1e-12, rtol=0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
