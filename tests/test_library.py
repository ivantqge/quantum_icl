"""Tests for the solution library."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from library import create_starter_library
from circuit import verify_circuit


def test_starter_library_verifies():
    lib = create_starter_library()
    assert lib.size() == 3
    for ex in lib.examples:
        passed, fidelity = verify_circuit(ex.task, ex.circuit)
        assert passed, f"{ex.task.task_id} failed: fidelity={fidelity}"
        assert fidelity > 0.999


def test_retrieve_returns_examples():
    lib = create_starter_library()
    task = lib.examples[0].task
    retrieved = lib.retrieve(task, k=2)
    assert 1 <= len(retrieved) <= 3
