"""Tests for Phase 2: unitary-synthesis tasks, gate sets, and verification."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit import CircuitDefinition, circuit_unitary, parse_circuit_json, verify_circuit
from unitary_task import (
    UnitarySynthesisTask,
    generate_unitary_tasks,
    create_unitary_starter_library,
)
from task import task_from_dict


def _bell_circuit():
    return CircuitDefinition(
        num_qubits=2,
        gates=[{"gate": "H", "qubits": [0]}, {"gate": "CNOT", "qubits": [0, 1]}],
    )


def _bell_task():
    circ = _bell_circuit()
    return UnitarySynthesisTask(
        num_qubits=2,
        target_unitary=circuit_unitary(circ),
        gate_set="clifford_t",
        generator_gate_count=2,
        generator_circuit=circ,
    )


def test_verify_exact_match():
    task = _bell_task()
    passed, fidelity = verify_circuit(task, _bell_circuit())
    assert passed
    assert fidelity == pytest.approx(1.0, abs=1e-6)


def test_verify_rejects_perturbed_circuit():
    task = _bell_task()
    wrong = CircuitDefinition(
        num_qubits=2,
        gates=[{"gate": "H", "qubits": [0]}, {"gate": "T", "qubits": [1]}],
    )
    passed, fidelity = verify_circuit(task, wrong)
    assert not passed
    assert fidelity < 0.999


def test_verify_is_global_phase_invariant():
    circ = _bell_circuit()
    phased = np.exp(1j * 0.789) * circuit_unitary(circ)
    task = UnitarySynthesisTask(
        num_qubits=2, target_unitary=phased, gate_set="clifford_t",
        generator_gate_count=2, generator_circuit=circ,
    )
    passed, fidelity = verify_circuit(task, circ)
    assert passed
    assert fidelity == pytest.approx(1.0, abs=1e-6)


def test_task_dict_round_trip():
    task = _bell_task()
    restored = task_from_dict(task.to_dict())
    assert restored.kind() == "unitary"
    assert restored.num_qubits == 2
    assert np.allclose(restored.target_unitary, task.target_unitary)
    assert restored.gate_set == "clifford_t"


def test_parse_clifford_t_accepts_and_rejects():
    good = '```json\n{"num_qubits": 2, "gates": [{"gate": "T", "qubits": [0]}, {"gate": "CNOT", "qubits": [0, 1]}]}\n```'
    circ = parse_circuit_json(good, gate_set="clifford_t")
    assert len(circ.gates) == 2

    # CZ is not in the clifford_t gate set.
    bad_gate = '```json\n{"num_qubits": 2, "gates": [{"gate": "CZ", "qubits": [0, 1]}]}\n```'
    with pytest.raises(ValueError):
        parse_circuit_json(bad_gate, gate_set="clifford_t")

    # CNOT requires exactly 2 qubits.
    bad_arity = '```json\n{"num_qubits": 2, "gates": [{"gate": "CNOT", "qubits": [0]}]}\n```'
    with pytest.raises(ValueError):
        parse_circuit_json(bad_arity, gate_set="clifford_t")


def test_unitary_starter_library_verifies():
    lib = create_unitary_starter_library("clifford_t")
    assert lib.size() == 4
    for ex in lib.examples:
        passed, fidelity = verify_circuit(ex.task, ex.circuit)
        assert passed, f"{ex.task.task_id} failed: fidelity={fidelity}"


def test_generate_unitary_tasks():
    import random
    tasks = generate_unitary_tasks(
        8, qubit_range=(1, 3), gate_set="clifford_t",
        gate_count_range=(2, 5), rng=random.Random(0),
    )
    assert len(tasks) == 8
    for t in tasks:
        assert t.kind() == "unitary"
        assert t.target_unitary.shape == (2 ** t.num_qubits, 2 ** t.num_qubits)
        # The generator circuit must reproduce the target unitary.
        passed, _ = verify_circuit(t, t.generator_circuit)
        assert passed
