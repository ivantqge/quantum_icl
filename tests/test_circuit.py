"""Tests for circuit metrics and simulation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit import CircuitDefinition, circuit_metrics, simulate_circuit


def _four_qubit_line():
    """The 4-qubit line graph state circuit: 4 H gates then 3 CZ gates."""
    return CircuitDefinition(
        num_qubits=4,
        gates=[
            {"gate": "H", "qubits": [0]},
            {"gate": "H", "qubits": [1]},
            {"gate": "H", "qubits": [2]},
            {"gate": "H", "qubits": [3]},
            {"gate": "CZ", "qubits": [0, 1]},
            {"gate": "CZ", "qubits": [1, 2]},
            {"gate": "CZ", "qubits": [2, 3]},
        ],
    )


def test_circuit_metrics_gate_counts():
    m = circuit_metrics(_four_qubit_line())
    assert m["gate_count"] == 7
    assert m["two_qubit_gate_count"] == 3
    assert m["num_qubits_used"] == 4


def test_circuit_metrics_depth():
    # 4 H gates pack into one moment; the 3 CZ gates each need their own
    # moment because consecutive CZ gates share a qubit -> depth 4.
    m = circuit_metrics(_four_qubit_line())
    assert m["depth"] == 4


def test_simulate_circuit_shape():
    state = simulate_circuit(_four_qubit_line())
    assert state.shape == (2 ** 4,)
