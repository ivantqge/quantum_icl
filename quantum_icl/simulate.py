"""Circuit simulation, fidelity measures, and circuit-quality metrics."""

import itertools

import numpy as np
import cirq

from .schema import GATE_ARITY

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

# gate name -> callable(qubits) -> cirq operation
_GATE_OPS = {
    "H": lambda q: cirq.H(q[0]),
    "S": lambda q: cirq.S(q[0]),
    "T": lambda q: cirq.T(q[0]),
    "X": lambda q: cirq.X(q[0]),
    "Y": lambda q: cirq.Y(q[0]),
    "Z": lambda q: cirq.Z(q[0]),
    "CX": lambda q: cirq.CNOT(q[0], q[1]),
    "CZ": lambda q: cirq.CZ(q[0], q[1]),
}


def build_circuit(circ: dict) -> cirq.Circuit:
    """Build a cirq.Circuit from a validated circuit dict."""
    qubits = cirq.LineQubit.range(circ["num_qubits"])
    ops = []
    for g in circ["gates"]:
        qs = [qubits[i] for i in g["qubits"]]
        ops.append(_GATE_OPS[g["gate"]](qs))
    return cirq.Circuit(ops)


def state_vector(circ: dict) -> np.ndarray:
    """Final state vector from |0...0> for a circuit dict."""
    cc = build_circuit(circ)
    qubits = cirq.LineQubit.range(circ["num_qubits"])
    result = cirq.Simulator().simulate(cc, qubit_order=qubits)
    return result.final_state_vector


def circuit_unitary(circ: dict) -> np.ndarray:
    """2^n x 2^n unitary of a circuit dict (all qubits included)."""
    qubits = cirq.LineQubit.range(circ["num_qubits"])
    cc = build_circuit(circ)
    return cc.unitary(qubit_order=qubits, qubits_that_should_be_present=qubits)


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """|<a|b>| for two pure-state vectors (global-phase invariant)."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.shape != b.shape:
        return 0.0
    return float(abs(np.vdot(a, b)))


def process_fidelity(u: np.ndarray, v: np.ndarray, num_qubits: int) -> float:
    """|Tr(u^dagger v)| / 2^n -- global-phase-invariant process fidelity.

    Equals 1 iff u and v are equal up to a global phase.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    if u.shape != v.shape:
        return 0.0
    d = 2 ** num_qubits
    return float(abs(np.trace(u.conj().T @ v)) / d)


def _pauli_matrix(label: str) -> np.ndarray:
    m = np.array([[1.0 + 0j]])
    for ch in label:
        m = np.kron(m, _PAULI[ch])
    return m


def stabilizer_generators(state, num_qubits: int) -> list:
    """Return n signed Pauli-string generators of a stabilizer state.

    Each generator is a string like "+XZ" or "-ZI": a Pauli operator that fixes
    the state (eigenvalue +/-1). Found by collecting non-identity Paulis with
    |<psi|P|psi>| ~ 1 and taking an independent set over GF(2). For small n
    (the stabilizer tier) the 4^n enumeration is cheap.
    """
    psi = np.asarray(state).reshape(-1)
    basis = {}   # leading bit -> reduced symplectic vector
    gens = []

    def symplectic(label):
        x = z = 0
        for i, ch in enumerate(label):
            if ch in ("X", "Y"):
                x |= 1 << i
            if ch in ("Z", "Y"):
                z |= 1 << i
        return x | (z << num_qubits)

    def try_add(v):
        while v:
            h = v.bit_length() - 1
            if h in basis:
                v ^= basis[h]
            else:
                basis[h] = v
                return True
        return False

    for combo in itertools.product("IXYZ", repeat=num_qubits):
        label = "".join(combo)
        if label == "I" * num_qubits:
            continue
        exp = np.vdot(psi, _pauli_matrix(label) @ psi).real
        if abs(abs(exp) - 1.0) < 1e-6 and try_add(symplectic(label)):
            gens.append(("+" if exp > 0 else "-") + label)
            if len(gens) == num_qubits:
                break
    return gens


def state_features(state, num_qubits: int) -> dict:
    """Structural features computable from a pure state vector alone (no hidden info)."""
    psi = np.asarray(state).reshape(-1).astype(complex)
    probs = (psi.conj() * psi).real
    amps = np.abs(psi)
    nonzero = int(np.sum(amps > 1e-9))
    sparsity = 1.0 - nonzero / len(psi)
    top = float(np.max(amps))
    p = probs + 1e-12
    entropy = float(-np.sum(p * np.log(p)))
    return {
        "num_qubits": num_qubits,
        "state_sparsity": float(sparsity),
        "state_nonzero": nonzero,
        "state_top_amp": top,
        "state_entropy": entropy,
    }


def unitary_features(U, num_qubits: int) -> dict:
    """Structural features computable from a unitary matrix alone."""
    U = np.asarray(U)
    d = 2 ** num_qubits
    A = np.abs(U)
    nonzero = int(np.sum(A > 1e-9))
    sparsity = 1.0 - nonzero / (d * d)
    diag_mass = float(np.sum(np.abs(np.diag(U)) ** 2) / d)
    frob_id = float(np.linalg.norm(U - np.eye(d)))
    # Sorted phase tuple of eigenvalues (rotation invariant within tier).
    eigs = np.linalg.eigvals(U)
    phases = sorted(float(np.angle(e)) for e in eigs)
    return {
        "num_qubits": num_qubits,
        "unitary_sparsity": float(sparsity),
        "unitary_diag_mass": diag_mass,
        "unitary_frob_to_identity": frob_id,
        "unitary_phases": phases,
    }


def circuit_metrics(circ: dict) -> dict:
    """Depth (cirq moments), gate count, two-qubit count, T-count."""
    gates = circ["gates"]
    gate_count = len(gates)
    two_qubit = sum(1 for g in gates if GATE_ARITY.get(g["gate"], 1) == 2)
    t_count = sum(1 for g in gates if g["gate"] == "T")
    depth = len(build_circuit(circ)) if gates else 0
    return {
        "depth": depth,
        "gate_count": gate_count,
        "two_qubit_gate_count": two_qubit,
        "t_count": t_count,
    }
