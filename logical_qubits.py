"""Named patch wiring, shared logical states, and continuing memory sessions."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from math import prod

import numpy as np

from simulator import PauliFrame, PauliString, Statevector, _integer
from surface_code import (
    PREPARATION,
    CircuitExtractorD3,
    CircuitFrameUpdate,
    CircuitHistoryDecoderD3,
    CircuitOperation,
    CircuitRound,
    CircuitScheduleD3,
    FaultLocation,
    RotatedSurfaceCodeD3,
    SyndromeExtractorD3,
    SyndromeRound,
    _data_state,
    circuit_operations,
    clean_readout,
)

# Patch wiring and shared extraction


def _patch_name(name):
    if not isinstance(name, str) or not name.strip() or "/" in name:
        raise ValueError("patch name must be nonempty and cannot contain '/'")
    return name


@dataclass(frozen=True)
class PatchOperation(CircuitOperation):
    """Mapped instruction with an explicit, patch-scoped classical register.

    Measurement writes record_key. Feedback reads that key and applies X
    only when its reported value is one. Other gates have no record key.
    """

    patch_name: str
    record_key: str | None = None


@dataclass(frozen=True)
class PatchSyndromeRound:
    """Ideal patch readout, retaining the entire shared physical state."""

    patch_name: str
    state: Statevector
    syndrome: tuple[int, ...]
    conditional_probabilities: tuple[float, ...]

    @property
    def probability(self):
        return prod(self.conditional_probabilities)


@dataclass(frozen=True)
class SurfaceCodePatchD3:
    """A named logical-qubit port: nine ordered data wires and one ancilla.

    This immutable handle owns wiring only, not a private wavefunction. Future
    operations can use the mapped instructions on a shared backend without
    losing inter-patch entanglement. The schedule uses local data positions;
    global wire indices need not be contiguous or sorted.
    """

    name: str
    data_qubits: tuple[int, ...]
    ancilla: int
    schedule: CircuitScheduleD3 = CircuitScheduleD3()

    def __post_init__(self):
        _patch_name(self.name)
        try:
            data = tuple(_integer(q, "data qubit") for q in self.data_qubits)
        except TypeError:
            raise ValueError("data_qubits must contain nine wire indices") from None
        ancilla = _integer(self.ancilla, "ancilla")
        if len(data) != 9 or min((*data, ancilla)) < 0 or len(set((*data, ancilla))) != 10:
            raise ValueError("patch requires nine distinct data wires and a distinct ancilla")
        if not isinstance(self.schedule, CircuitScheduleD3):
            raise ValueError("schedule must be CircuitScheduleD3")
        object.__setattr__(self, "data_qubits", data)
        object.__setattr__(self, "ancilla", ancilla)

    @classmethod
    def contiguous(cls, name: str, offset: int = 0, *, schedule=None) -> "SurfaceCodePatchD3":
        offset = _integer(offset, "offset")
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        return cls(
            name,
            tuple(range(offset, offset + 9)),
            offset + 9,
            CircuitScheduleD3() if schedule is None else schedule,
        )

    @property
    def qubits(self) -> tuple[int, ...]:
        """Local-to-global wire map: data positions 0..8, then ancilla position 9."""
        return (*self.data_qubits, self.ancilla)

    def global_qubit(self, local_qubit: int) -> int:
        local = _integer(local_qubit, "local_qubit")
        if not 0 <= local < 10:
            raise ValueError("local_qubit must be in 0..9")
        return self.qubits[local]

    def _check_width(self, n_qubits):
        n = _integer(n_qubits, "n_qubits")
        if n <= max(self.qubits):
            raise ValueError("state width must include every patch wire")
        return n

    def lift_pauli(self, pauli: PauliString, n_qubits: int) -> PauliString:
        """Embed a local nine-data or ten-wire Pauli into a shared register."""
        n = self._check_width(n_qubits)
        if not isinstance(pauli, PauliString) or pauli.n_qubits not in (9, 10):
            raise ValueError("local Pauli must have nine or ten qubits")
        labels = ["I"] * n
        for local, axis in enumerate(pauli.labels):
            labels[self.qubits[local]] = axis
        return PauliString("".join(labels), pauli.sign)

    def lift_frame(self, frame: PauliFrame, n_qubits: int) -> PauliFrame:
        """Embed a local correction without mutating it or the shared state."""
        if not isinstance(frame, PauliFrame) or frame.n_qubits not in (9, 10):
            raise ValueError("local frame must have nine or ten qubits")
        return PauliFrame.from_pauli(self.lift_pauli(frame.pauli, n_qubits))

    def logical_pauli(self, axis: str, n_qubits: int) -> PauliString:
        """Global logical X/Y/Z observable; this does not implement a gate protocol."""
        code = RotatedSurfaceCodeD3()
        if axis == "X":
            local = code.logical_x
        elif axis == "Z":
            local = code.logical_z
        elif axis == "Y":
            # These fixed X/Z representatives overlap once: Y_L = i X_L Z_L.
            local = PauliFrame.from_pauli(code.logical_x).updated(code.logical_z).pauli
        else:
            raise ValueError("logical axis must be X, Y, or Z")
        return self.lift_pauli(local, n_qubits)

    def stabilizers(self, n_qubits: int) -> tuple[PauliString, ...]:
        return tuple(self.lift_pauli(check, n_qubits) for check in RotatedSurfaceCodeD3.stabilizers)

    def _map_operation(self, operation):
        prefix = self.name + "/"
        record_key = None
        if operation.gate in ("measure", "feedback"):
            record_key = prefix + RotatedSurfaceCodeD3.stabilizer_names[operation.check]
        return PatchOperation(
            prefix + operation.name,
            operation.gate,
            tuple(self.global_qubit(q) for q in operation.targets),
            operation.check,
            tuple(
                FaultLocation(
                    prefix + location.name,
                    location.rate,
                    tuple(self.global_qubit(q) for q in location.targets),
                    location.alternatives,
                )
                for location in operation.locations
            ),
            self.name,
            record_key,
        )

    def preparation_operation(self) -> PatchOperation:
        """One initial preparation slot; do not append before every round."""
        return self._map_operation(PREPARATION)

    def round_operations(self) -> tuple[PatchOperation, ...]:
        """48 remapped slots and scoped fault locations for a future scheduler.

        Idle locations cover this patch's inactive wires only. A register-wide
        scheduler must account for other patches' idle time and any overlaps.
        """
        return tuple(self._map_operation(op) for op in circuit_operations(self.schedule))

    def measure_ideal_round(self, state: Statevector, rng=None) -> PatchSyndromeRound:
        """Ideal extraction on a shared state, preserving other patches and entanglement.

        The selected ancilla must start in |0>. It is actually measured/reset;
        no local pure-state factorization is assumed or performed. This method
        samples no faults and grants no multi-patch fault-tolerance certificate.
        """
        self._check_width(state.n_qubits)
        ancilla_one = state.amplitudes.reshape(-1, 2, 1 << self.ancilla)[:, 1, :]
        if np.any(ancilla_one != 0):
            raise ValueError("selected patch ancilla must be reset to |0>")
        joint = state.copy()
        generator = np.random.default_rng() if rng is None else rng
        bits, probabilities = [], []
        observed = 0
        observable = self.lift_pauli(PauliString("I" * 9 + "Z"), state.n_qubits)
        for operation in self.round_operations():
            if operation.gate == "h":
                joint.h(*operation.targets)
            elif operation.gate == "cnot":
                joint.cnot(*operation.targets)
            elif operation.gate == "measure":
                p = (joint.probability(observable, 1), joint.probability(observable, -1))
                observed = joint.measure_z(self.ancilla, generator)
                bits.append(observed)
                probabilities.append(p[observed])
            elif operation.gate == "feedback" and observed:
                joint.x(self.ancilla)
        return PatchSyndromeRound(self.name, joint, tuple(bits), tuple(probabilities))

    def start_memory(self, initial="0", **kwargs) -> "PatchMemoryD3":
        """Create an independent local memory session with this patch's identity.

        The session is a one-patch simulator. It is not a view of a shared
        entangled register; use mapped operations/shared-state extraction there.
        """

        return PatchMemoryD3(self, initial, **kwargs)


# Logical register layout and encoding


@lru_cache(maxsize=1)
def _encoding_basis():
    code = RotatedSurfaceCodeD3()
    basis = np.zeros((1024, 2), dtype=np.complex128)
    basis[:512, 0] = code.logical_zero().amplitudes
    basis[:512, 1] = code.logical_one().amplitudes
    basis.flags.writeable = False
    return basis


def _logical_input(initial):
    if isinstance(initial, str):
        states = {
            "0": [1, 0],
            "1": [0, 1],
            "+": [1, 1],
            "-": [1, -1],
            "+i": [1, 1j],
            "-i": [1, -1j],
        }
        if initial not in states:
            raise ValueError("initial label must be 0, 1, +, -, +i, or -i")
        amplitudes = np.array(states[initial], dtype=np.complex128)
        return Statevector(amplitudes / np.linalg.norm(amplitudes))
    if not isinstance(initial, Statevector) or initial.n_qubits != 1:
        raise ValueError("initial must be a logical label or a one-qubit Statevector")
    return initial.copy()


class LogicalRegisterD3:
    """Immutable layout: logical wire i owns physical wires 10*i .. 10*i+9.

    Construction allocates only handles. encode/decode explicitly use the dense
    backend and support arbitrary logical entanglement. Large layouts can still
    be passed to a future scalable backend without allocating a wavefunction.
    """

    def __init__(self, names: Iterable[str], *, schedule: CircuitScheduleD3 | None = None):
        if isinstance(names, str):
            raise ValueError("names must be a sequence of patch names, not one string")
        try:
            patches = tuple(
                SurfaceCodePatchD3.contiguous(name, 10 * i, schedule=schedule)
                for i, name in enumerate(names)
            )
        except TypeError:
            raise ValueError("names must be an iterable of patch names") from None
        if not patches or len({patch.name for patch in patches}) != len(patches):
            raise ValueError("register needs at least one patch and unique names")
        self._patches = patches
        self._by_name = {patch.name: patch for patch in patches}

    @property
    def patches(self) -> tuple[SurfaceCodePatchD3, ...]:
        return self._patches

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(patch.name for patch in self.patches)

    @property
    def n_logical(self) -> int:
        return len(self.patches)

    @property
    def n_physical(self) -> int:
        return 10 * self.n_logical

    def __len__(self):
        return self.n_logical

    def __getitem__(self, name: str) -> SurfaceCodePatchD3:
        """Resolve a logical wire by its stable name (not a physical index)."""
        return self._by_name[name]

    def _check_budget(self, max_qubits):
        maximum = _integer(max_qubits, "max_qubits")
        if maximum < 1:
            raise ValueError("max_qubits must be positive")
        if self.n_physical > maximum:
            raise ValueError(
                f"dense encoding needs {self.n_physical} physical qubits, exceeding "
                f"max_qubits={maximum}; the layout itself needs no statevector. "
                "Increase the explicit limit only if amplitude storage and temporaries fit."
            )

    def encode(self, logical_state: Statevector, *, max_qubits: int = 20) -> Statevector:
        """Ideal shared encoding, preserving arbitrary logical entanglement.

        Logical bit i maps to patches[i], with q0 least significant. Ancillas
        start at |0>. The default limit permits two patches (16 MiB amplitudes)
        and refuses three (16 GiB) before allocating any physical state.
        """
        self._check_budget(max_qubits)
        if not isinstance(logical_state, Statevector) or logical_state.n_qubits != self.n_logical:
            raise ValueError("logical state width must equal the register's logical-qubit count")
        tensor = logical_state.amplitudes.reshape((2,) * self.n_logical)
        basis = _encoding_basis()
        for axis in range(self.n_logical):
            tensor = np.tensordot(basis, tensor, axes=(1, axis))
            tensor = np.moveaxis(tensor, 0, axis)
        return Statevector(tensor.reshape(-1))

    def encode_product(
        self, initial: Mapping[str, str | Statevector], *, max_qubits: int = 20
    ) -> Statevector:
        """Convenience boundary for named product inputs; all names are required."""
        self._check_budget(max_qubits)
        if not isinstance(initial, Mapping) or set(initial) != set(self.names):
            raise ValueError("product inputs must specify every patch name exactly once")
        vector = np.array([1], dtype=np.complex128)
        for name in self.names:
            vector = np.kron(_logical_input(initial[name]).amplitudes, vector)
        return self.encode(Statevector(vector), max_qubits=max_qubits)

    def decode(
        self, physical_state: Statevector, *, frame: PauliFrame | None = None
    ) -> Statevector:
        """Ideal inverse boundary; reject syndromes/unreset ancillas, never postselect.

        Supply a known shared frame to account for pending corrections. This is
        an isometry check/export utility, not a noisy logical readout protocol.
        """
        if (
            not isinstance(physical_state, Statevector)
            or physical_state.n_qubits != self.n_physical
        ):
            raise ValueError("physical state width must equal the register layout")
        if frame is not None and not isinstance(frame, PauliFrame):
            raise ValueError("frame must be a shared PauliFrame")
        state = physical_state if frame is None else frame.apply(physical_state)
        tensor = state.amplitudes.reshape((1024,) * self.n_logical)
        inverse = _encoding_basis().conj().T
        for axis in range(self.n_logical):
            tensor = np.tensordot(inverse, tensor, axes=(1, axis))
            tensor = np.moveaxis(tensor, 0, axis)
        vector = tensor.reshape(-1)
        norm = float(np.vdot(vector, vector).real)
        if not np.isclose(norm, 1, atol=1e-12, rtol=0):
            raise ValueError("state must be in every patch's codespace with ancillas reset to |0>")
        # Correct only floating-point drift after the isometry membership check.
        return Statevector(vector / np.sqrt(norm))

    def combine_frames(self, frames: Mapping[str, PauliFrame]) -> PauliFrame:
        """Lift named local data frames; omitted patches have identity frames."""
        if not isinstance(frames, Mapping) or not set(frames) <= set(self.names):
            raise ValueError("frames must refer only to names in this register")
        combined = PauliFrame(self.n_physical)
        for name, frame in frames.items():
            if not isinstance(frame, PauliFrame) or frame.n_qubits != 9:
                raise ValueError("each patch correction must be a nine-data-qubit PauliFrame")
            lifted = self[name].lift_frame(frame, self.n_physical)
            combined = combined.updated(lifted.pauli)
        return combined


# Continuing local memory


@dataclass(frozen=True)
class PatchMemoryRecord:
    """Patch-tagged local readout and classical recovery decision.

    State snapshots use local data 0..8 and ancilla 9, regardless of the patch's
    external mapping. Fault diagnostics retain local CircuitFault identifiers.
    """

    patch_name: str
    measurement: CircuitRound | SyndromeRound
    update: CircuitFrameUpdate

    @property
    def syndrome(self):
        return self.measurement.syndrome

    @property
    def frame(self):
        return self.update.frame


class PatchMemoryD3:
    """Independent physical patch session, advanced one complete round at a time.

    Use this for local memory experiments. Separate sessions cannot represent
    entangled logical qubits; those require one shared register state/backend.
    No interleaved external gates or changes to the noise prior are accepted.
    """

    def __init__(
        self,
        patch: SurfaceCodePatchD3,
        initial="0",
        *,
        noise=None,
        rng=None,
        initial_frame: PauliFrame | None = None,
        certification_rounds=3,
        preparation_injections=(),
        sample_noise=True,
    ):
        if not isinstance(patch, SurfaceCodePatchD3):
            raise ValueError("patch must be SurfaceCodePatchD3")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        extractor = CircuitExtractorD3(noise, patch.schedule)
        decoder = CircuitHistoryDecoderD3(
            extractor.noise,
            patch.schedule,
            initial_frame,
            certification_rounds=certification_rounds,
        )
        code = RotatedSurfaceCodeD3()
        if isinstance(initial, Statevector) and initial.n_qubits == 9:
            state = initial.copy()
            if not code.in_codespace(state):
                raise ValueError("initial nine-qubit data must be in the clean codespace")
        else:
            state = code.encode(*_logical_input(initial).amplitudes)
        # initial_frame specifies a known Pauli to APPLY to the clean input.
        # To continue an existing trajectory, retain this session instead.
        state = decoder.frame.apply(state)
        self._rng = np.random.default_rng() if rng is None else rng
        preparation = extractor.prepare(
            state, self._rng, injections=preparation_injections, sample_noise=sample_noise
        )
        self._patch, self._extractor, self._decoder = patch, extractor, decoder
        self._joint = preparation.joint_state
        self._preparation_faults = preparation.faults
        self._records = []
        self._final_record = None

    @property
    def patch(self):
        return self._patch

    @property
    def noise(self):
        return self._extractor.noise

    @property
    def n_rounds(self):
        return self._decoder.n_rounds

    @property
    def closed(self):
        return self._final_record is not None

    @property
    def frame(self):
        return self._decoder.frame

    @property
    def posterior(self):
        return self._decoder.posterior

    @property
    def state(self):
        """Owned local nine-data-qubit snapshot, with pending frame unapplied."""
        return _data_state(self._joint)

    @property
    def joint_state(self):
        """Owned local ten-qubit snapshot; never discard the internal ancilla."""
        return self._joint.copy()

    @property
    def corrected_state(self):
        return self.frame.apply(self.state)

    @property
    def records(self):
        return tuple(self._records)

    @property
    def history(self):
        return tuple(record.syndrome for record in self._records)

    @property
    def final_record(self):
        return self._final_record

    @property
    def preparation_faults(self):
        return self._preparation_faults

    def step(self, *, injections=(), sample_noise=True) -> PatchMemoryRecord:
        """Execute the next round; injection indices are absolute within this session.

        Preparation happens only at construction. Recovery sees reported bits
        only. Commit physical state only after the decoder accepts the record.
        """
        if self.closed:
            raise ValueError("memory is closed; start a new session for a new experiment")
        measured = self._extractor.extract_round(
            self._joint,
            self._rng,
            round_index=self.n_rounds,
            injections=injections,
            sample_noise=sample_noise,
        )
        update = self._decoder.update(measured.syndrome)
        record = PatchMemoryRecord(self.patch.name, measured, update)
        self._joint = measured.joint_state.copy()
        self._records.append(record)
        return record

    def finish(self) -> PatchMemoryRecord:
        """Actually perform the explicit ideal reset/readout boundary, then close.

        This is optional, not an automatic cleanup between rounds. It is the
        same boundary assumption used by CircuitFrameCorrectionD3.run.
        """
        if self.closed or self.n_rounds == 0:
            raise ValueError("finish requires at least one round and an open memory")
        measured = clean_readout(self._joint, self._rng)
        update = self._decoder.finish(measured.syndrome)
        self._joint = SyndromeExtractorD3().prepare(measured.state)
        self._final_record = PatchMemoryRecord(self.patch.name, measured, update)
        return self._final_record
