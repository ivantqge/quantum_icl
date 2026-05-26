"""Circuit simulation, fidelity measures, and circuit-quality metrics."""

import numpy as np
import cirq

from .schema import GATE_ARITY

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
