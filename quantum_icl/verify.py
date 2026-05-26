"""Correctness verifier: schema -> simulate -> compare to target.

Rejects invalid JSON, unsupported/disallowed gates, and invalid qubit indices;
simulates the circuit and compares to the task target up to global phase.
Records fidelity, success, and circuit-quality metrics.
"""

from dataclasses import dataclass, field

from .schema import extract_json, validate_circuit_dict, SchemaError
from .simulate import (
    state_vector, circuit_unitary, state_fidelity, process_fidelity,
    circuit_metrics,
)


@dataclass
class VerifyResult:
    valid: bool
    success: bool
    fidelity: float
    error: str = ""
    circuit: dict = None
    metrics: dict = field(default_factory=dict)


def verify(task, response, fidelity_threshold: float = 0.999) -> VerifyResult:
    """Verify an LLM response (text or circuit dict) against a task target."""
    # 1. Parse + schema-validate, enforcing the tier's gate set.
    try:
        circ = extract_json(response)
        validate_circuit_dict(circ, allowed_gates=task.gate_set)
    except SchemaError as e:
        return VerifyResult(valid=False, success=False, fidelity=0.0, error=str(e))

    # 2. Qubit count must match the target dimension.
    if circ["num_qubits"] != task.num_qubits:
        return VerifyResult(
            valid=True, success=False, fidelity=0.0, circuit=circ,
            error=(f"num_qubits {circ['num_qubits']} != target "
                   f"{task.num_qubits}"),
            metrics=circuit_metrics(circ),
        )

    # 3. Simulate and compare (global-phase invariant).
    try:
        if task.target_kind == "state":
            fid = state_fidelity(task.target, state_vector(circ))
        else:
            fid = process_fidelity(task.target, circuit_unitary(circ),
                                   task.num_qubits)
    except Exception as e:  # numerical / construction failure
        return VerifyResult(valid=True, success=False, fidelity=0.0,
                            circuit=circ, error=f"simulation failed: {e}",
                            metrics=circuit_metrics(circ))

    return VerifyResult(
        valid=True,
        success=fid > fidelity_threshold,
        fidelity=fid,
        circuit=circ,
        metrics=circuit_metrics(circ),
    )
