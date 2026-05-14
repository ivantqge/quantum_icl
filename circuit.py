"""Circuit definition, JSON parsing, simulation, and verification."""

from dataclasses import dataclass
import json
import re
import numpy as np
import cirq

from graph_task import GraphStateTask

# Gate-set registry: each name maps to its allowed gate symbols.
GATE_SETS = {
    "graph_state": {"H", "CZ"},
    "clifford_t": {"H", "S", "T", "CNOT"},
}

# Number of qubits each gate symbol acts on.
_GATE_ARITY = {"H": 1, "S": 1, "T": 1, "CZ": 2, "CNOT": 2}

# Backward-compatible alias for the Phase 1 graph-state gate set.
ALLOWED_GATES = GATE_SETS["graph_state"]


@dataclass
class CircuitDefinition:
    """A quantum circuit as a list of gates."""
    num_qubits: int
    gates: list  # list of {"gate": str, "qubits": list[int]}

    def to_dict(self) -> dict:
        return {"num_qubits": self.num_qubits, "gates": self.gates}

    @classmethod
    def from_dict(cls, d: dict) -> "CircuitDefinition":
        return cls(num_qubits=d["num_qubits"], gates=d["gates"])

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def parse_circuit_json(
    text: str, gate_set: str = "graph_state"
) -> CircuitDefinition:
    """Extract and parse a circuit JSON from LLM response text.

    Handles markdown code fences, trailing commas, and other common issues.
    `gate_set` selects which gate symbols are allowed (see GATE_SETS).
    """
    # Try to find JSON in code fences first
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        json_str = fence_match.group(1)
    else:
        # Find the first { ... } block containing "gates"
        brace_depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == "{":
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0 and start is not None:
                    candidate = text[start : i + 1]
                    if "gates" in candidate:
                        json_str = candidate
                        break
        else:
            raise ValueError("No valid circuit JSON found in response")

    # Clean up common issues
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)  # trailing commas
    json_str = re.sub(r"//.*$", "", json_str, flags=re.MULTILINE)  # line comments

    data = json.loads(json_str)

    if "num_qubits" not in data or "gates" not in data:
        raise ValueError("JSON missing required fields 'num_qubits' and/or 'gates'")

    allowed = GATE_SETS.get(gate_set)
    if allowed is None:
        raise ValueError(f"Unknown gate set '{gate_set}'")

    # Validate gates
    for g in data["gates"]:
        gate_name = g.get("gate", "")
        qubits = g.get("qubits", [])

        if gate_name not in allowed:
            raise ValueError(f"Gate '{gate_name}' not in allowed set {allowed}")

        arity = _GATE_ARITY[gate_name]
        if len(qubits) != arity:
            raise ValueError(
                f"{gate_name} gate requires exactly {arity} qubit(s), "
                f"got {len(qubits)}"
            )

        for q in qubits:
            if not (0 <= q < data["num_qubits"]):
                raise ValueError(
                    f"Qubit index {q} out of range [0, {data['num_qubits']})"
                )

    return CircuitDefinition(num_qubits=data["num_qubits"], gates=data["gates"])


def _build_cirq_circuit(circuit: CircuitDefinition) -> cirq.Circuit:
    """Construct a cirq.Circuit from a CircuitDefinition.

    Shared by simulation and metric computation so there is a single
    gate-construction code path.
    """
    qubits = cirq.LineQubit.range(circuit.num_qubits)
    ops = []
    for g in circuit.gates:
        name = g["gate"]
        qs = [qubits[i] for i in g["qubits"]]
        if name == "H":
            ops.append(cirq.H(qs[0]))
        elif name == "S":
            ops.append(cirq.S(qs[0]))
        elif name == "T":
            ops.append(cirq.T(qs[0]))
        elif name == "CZ":
            ops.append(cirq.CZ(qs[0], qs[1]))
        elif name == "CNOT":
            ops.append(cirq.CNOT(qs[0], qs[1]))
    return cirq.Circuit(ops)


def simulate_circuit(circuit: CircuitDefinition) -> np.ndarray:
    """Simulate a circuit using Cirq and return the final state vector."""
    cirq_circuit = _build_cirq_circuit(circuit)
    result = cirq.Simulator().simulate(cirq_circuit)
    return result.final_state_vector


def circuit_unitary(circuit: CircuitDefinition) -> np.ndarray:
    """Return the 2^n x 2^n unitary matrix implemented by the circuit.

    All `num_qubits` qubits are included even if some are untouched, so the
    matrix dimension always matches the declared qubit count.
    """
    qubits = cirq.LineQubit.range(circuit.num_qubits)
    cirq_circuit = _build_cirq_circuit(circuit)
    return cirq_circuit.unitary(
        qubit_order=qubits,
        qubits_that_should_be_present=qubits,
    )


def circuit_metrics(circuit: CircuitDefinition) -> dict:
    """Compute circuit-quality metrics for logging and analysis.

    Returns gate_count, two_qubit_gate_count, depth (number of cirq moments),
    num_qubits_used (distinct qubit indices touched), and t_count.
    """
    gate_count = len(circuit.gates)
    two_qubit_gate_count = sum(
        1 for g in circuit.gates if len(g.get("qubits", [])) >= 2
    )
    t_count = sum(1 for g in circuit.gates if g.get("gate") == "T")
    used = set()
    for g in circuit.gates:
        used.update(g.get("qubits", []))
    depth = len(_build_cirq_circuit(circuit))
    return {
        "gate_count": gate_count,
        "two_qubit_gate_count": two_qubit_gate_count,
        "t_count": t_count,
        "depth": depth,
        "num_qubits_used": len(used),
    }


def verify_circuit(
    task,
    circuit: CircuitDefinition,
    threshold: float = 0.999,
) -> tuple:
    """Verify whether a circuit realizes a task's target.

    Returns (passed: bool, fidelity: float). Dispatches on task.kind():
    - "graph_state": state fidelity |<target|output>|.
    - "unitary": global-phase-invariant Hilbert-Schmidt process fidelity
      |Tr(U_target^dagger U_output)| / 2^n.
    Both equal 1 iff the target is matched up to global phase.
    """
    try:
        kind = task.kind() if hasattr(task, "kind") else "graph_state"
        if kind == "unitary":
            target = task.target()
            output = circuit_unitary(circuit)
            if output.shape != target.shape:
                return (False, 0.0)
            d = 2 ** task.num_qubits
            fidelity = float(abs(np.trace(target.conj().T @ output)) / d)
        else:
            target = task.target_state_vector()
            output = simulate_circuit(circuit)
            fidelity = float(abs(np.vdot(target, output)))
        return (fidelity > threshold, fidelity)
    except Exception as e:
        return (False, 0.0)
