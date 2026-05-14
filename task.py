"""Task abstraction shared by graph-state and unitary-synthesis tasks.

A `Task` is whatever the experiment pipeline needs to pose a synthesis problem
to the LLM and verify the answer. Both GraphStateTask and UnitarySynthesisTask
satisfy this protocol; `task_from_dict` rebuilds the right concrete type from a
serialized dict by dispatching on its "kind" field.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Task(Protocol):
    """Structural protocol for a synthesis task."""

    num_qubits: int
    task_id: str

    def kind(self) -> str:
        """Task kind: 'graph_state' or 'unitary'."""
        ...

    def description(self) -> str:
        """Human-readable description for LLM prompts."""
        ...

    def target(self):
        """Target object: a state vector (graph_state) or unitary (unitary)."""
        ...

    def gate_set_name(self) -> str:
        """Name of the allowed gate set (key into circuit.GATE_SETS)."""
        ...

    def structural_features(self) -> dict:
        """Structural features used for similarity-based retrieval."""
        ...

    def to_dict(self) -> dict:
        ...


def task_from_dict(d: dict):
    """Rebuild a concrete task from a serialized dict, dispatching on 'kind'."""
    kind = d.get("kind", "graph_state")
    if kind == "graph_state":
        from graph_task import GraphStateTask
        return GraphStateTask.from_dict(d)
    if kind == "unitary":
        from unitary_task import UnitarySynthesisTask
        return UnitarySynthesisTask.from_dict(d)
    raise ValueError(f"Unknown task kind '{kind}'")
