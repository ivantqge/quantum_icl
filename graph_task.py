"""Graph state task definitions and generation."""

from dataclasses import dataclass, field
import random
import numpy as np
import cirq


@dataclass
class GraphStateTask:
    """A graph state preparation task on n qubits.

    A graph state |G> is defined by a graph G = (V, E) and prepared by:
    1. Apply H to all qubits (starting from |0...0>)
    2. Apply CZ to each edge (i, j) in E
    """
    num_qubits: int
    edges: list  # list of (int, int) tuples
    task_id: str = ""

    def __post_init__(self):
        if not self.task_id:
            edge_str = "_".join(f"{a}{b}" for a, b in sorted(self.edges))
            self.task_id = f"{self.num_qubits}q_edges_{edge_str}"

    def adjacency_matrix(self) -> np.ndarray:
        mat = np.zeros((self.num_qubits, self.num_qubits), dtype=int)
        for i, j in self.edges:
            mat[i][j] = 1
            mat[j][i] = 1
        return mat

    def degree_sequence(self) -> tuple:
        """Sorted degree sequence, used for similarity matching."""
        degrees = [0] * self.num_qubits
        for i, j in self.edges:
            degrees[i] += 1
            degrees[j] += 1
        return tuple(sorted(degrees))

    def target_state_vector(self) -> np.ndarray:
        """Compute exact target state by simulating the canonical circuit in Cirq."""
        qubits = cirq.LineQubit.range(self.num_qubits)
        ops = [cirq.H(q) for q in qubits]
        ops += [cirq.CZ(qubits[i], qubits[j]) for i, j in self.edges]
        circuit = cirq.Circuit(ops)
        result = cirq.Simulator().simulate(circuit)
        return result.final_state_vector

    # --- Task protocol adapters (see task.py) ---

    def kind(self) -> str:
        return "graph_state"

    def gate_set_name(self) -> str:
        return "graph_state"

    def target(self) -> np.ndarray:
        """Target object for verification: the graph state vector."""
        return self.target_state_vector()

    def structural_features(self) -> dict:
        return {
            "num_qubits": self.num_qubits,
            "edge_count": len(self.edges),
            "degree_sequence": list(self.degree_sequence()),
        }

    def to_dict(self) -> dict:
        return {
            "kind": "graph_state",
            "num_qubits": self.num_qubits,
            "edges": [list(e) for e in self.edges],
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GraphStateTask":
        return cls(
            num_qubits=d["num_qubits"],
            edges=[tuple(e) for e in d["edges"]],
            task_id=d.get("task_id", ""),
        )

    def description(self) -> str:
        """Human-readable description for LLM prompts."""
        lines = [f"Graph state on {self.num_qubits} qubits."]
        lines.append(f"Edges: {self.edges}")
        adj = self.adjacency_matrix()
        lines.append("Adjacency matrix:")
        for row in adj:
            lines.append("  " + " ".join(str(x) for x in row))
        return "\n".join(lines)


def generate_random_tasks(
    num_tasks: int,
    qubit_range: tuple = (4, 6),
    edge_prob: float = 0.5,
    rng: random.Random = None,
) -> list:
    """Generate random graph state tasks, deduplicated by edge set."""
    if rng is None:
        rng = random.Random()

    seen = set()
    tasks = []

    while len(tasks) < num_tasks:
        n = rng.randint(qubit_range[0], qubit_range[1])
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < edge_prob:
                    edges.append((i, j))
        if not edges:
            continue

        key = (n, tuple(sorted(edges)))
        if key in seen:
            continue
        seen.add(key)

        task = GraphStateTask(num_qubits=n, edges=edges)
        tasks.append(task)

    return tasks
