# ftqc-explicit

Surface-code error correction and reversible logical addition with NumPy.

```bash
python -m pip install numpy
python adder.py --bits 4 --a 11 --b 9
python check.py
```

- `simulator.py`: statevectors, Pauli operators, and correction frames.
- `surface_code.py`: the D3 patch, noise, extraction, decoders, and fault audits.
- `logical_qubits.py`: reusable patches and shared logical registers.
- `adder.py`: coherent addition and lightweight binary-input simulation.
- `check.py`: integration checks.

Python 3.11+. MIT license; notice in `simulator.py`.
