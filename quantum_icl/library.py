"""Verified-solution library: storage and serialization.

Only verified-correct circuits are added. Each entry keeps the task target,
structural features, the solution circuit, and its quality metrics so any
retrieval strategy can score it.
"""

import json

import numpy as np


def _serialize_target(target):
    t = np.asarray(target)
    return {"real": t.real.tolist(), "imag": t.imag.tolist()}


def make_entry(task, solution_circuit: dict, metrics: dict) -> dict:
    """Build a library entry from a solved task."""
    return {
        "task_id": task.task_id,
        "tier": task.tier,
        "num_qubits": task.num_qubits,
        "target_kind": task.target_kind,
        "target": _serialize_target(task.target),
        "features": task.features,
        "oracle_features": getattr(task, "oracle_features", {}) or {},
        "description": task.description,
        "gate_set": list(task.gate_set),
        "solution_circuit": solution_circuit,
        "metrics": metrics,
    }


class VerifiedLibrary:
    """Append-only store of verified solution entries."""

    def __init__(self, entries=None):
        self.entries = list(entries) if entries else []

    def add(self, task, solution_circuit: dict, metrics: dict):
        self.entries.append(make_entry(task, solution_circuit, metrics))

    def size(self) -> int:
        return len(self.entries)

    def all(self):
        return list(self.entries)

    def to_json(self) -> str:
        return json.dumps(self.entries, indent=2)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "VerifiedLibrary":
        with open(path) as f:
            return cls(json.load(f))
