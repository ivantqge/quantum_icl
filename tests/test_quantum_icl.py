"""Tests for the quantum_icl framework: schema, verification, tasks, retrieval."""

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_icl import tasks as T
from quantum_icl.schema import validate_circuit_dict, extract_json, SchemaError
from quantum_icl.verify import verify
from quantum_icl.library import VerifiedLibrary
from quantum_icl.retrieval import StructuralRetrieval, RandomRetrieval, TextRetrieval
from quantum_icl.simulate import circuit_unitary


# --- schema ---------------------------------------------------------------

def test_schema_rejects_bad_inputs():
    with pytest.raises(SchemaError):
        validate_circuit_dict({"gates": []})                       # no num_qubits
    with pytest.raises(SchemaError):
        validate_circuit_dict({"num_qubits": 1, "gates": [{"gate": "Q", "qubits": [0]}]})
    with pytest.raises(SchemaError):
        validate_circuit_dict({"num_qubits": 1, "gates": [{"gate": "CX", "qubits": [0]}]})
    with pytest.raises(SchemaError):
        validate_circuit_dict({"num_qubits": 1, "gates": [{"gate": "H", "qubits": [3]}]})


def test_schema_enforces_allowed_gates():
    circ = {"num_qubits": 2, "gates": [{"gate": "T", "qubits": [0]}]}
    validate_circuit_dict(circ, allowed_gates=["H", "S", "T", "CX"])
    with pytest.raises(SchemaError):
        validate_circuit_dict(circ, allowed_gates=["H", "S", "CX"])  # Clifford only


def test_extract_json_from_fenced_text():
    text = "Strategy: do it.\n```json\n{\"num_qubits\": 1, \"gates\": []}\n```"
    assert extract_json(text)["num_qubits"] == 1


# --- verification correctness (the scientific core) -----------------------

@pytest.mark.parametrize("tier", ["A", "B", "C", "D"])
def test_generator_solves_its_own_task(tier):
    rng = random.Random(0)
    task = T.generate_tasks(tier, 1, rng=rng)[0]
    res = verify(task, task.generator)
    assert res.valid and res.success
    assert res.fidelity == pytest.approx(1.0, abs=1e-6)


def test_perturbed_circuit_fails():
    task = T.generate_tasks("D", 1, rng=random.Random(1))[0]
    bad = {"num_qubits": task.num_qubits,
           "gates": task.generator["gates"] + [{"gate": "H", "qubits": [0]}]}
    res = verify(task, bad)
    assert not res.success
    assert res.fidelity < 0.999


def test_process_fidelity_global_phase_invariant():
    task = T.generate_tasks("C", 1, rng=random.Random(2))[0]
    # Apply a global phase to the target; the generator must still verify.
    task.target = np.exp(1j * 1.2345) * task.target
    res = verify(task, task.generator)
    assert res.success and res.fidelity == pytest.approx(1.0, abs=1e-6)


def test_tier_gate_set_enforced_in_verify():
    # A Tier C (Clifford) task must reject a solution that uses T.
    task = T.generate_tasks("C", 1, rng=random.Random(3))[0]
    sneaky = {"num_qubits": task.num_qubits, "gates": [{"gate": "T", "qubits": [0]}]}
    res = verify(task, sneaky)
    assert not res.valid


# --- retrieval ------------------------------------------------------------

def test_retrieval_same_tier_only_and_topk():
    rng = random.Random(4)
    a_tasks = T.generate_tasks("A", 6, rng=rng)
    lib = VerifiedLibrary()
    for t in a_tasks:
        lib.add(t, t.generator, {"depth": 1, "gate_count": 1,
                                 "two_qubit_gate_count": 0, "t_count": 0})
    query = a_tasks[0]
    for retr in (StructuralRetrieval(), RandomRetrieval(random.Random(0)), TextRetrieval()):
        sel = retr.select(query, lib, k=3)
        assert len(sel) == 3
        assert all(e["tier"] == "A" for e in sel)
