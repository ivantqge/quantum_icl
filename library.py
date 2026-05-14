"""Solution library: storage, retrieval, and starter examples."""

from dataclasses import dataclass
import json
import random

from graph_task import GraphStateTask
from circuit import CircuitDefinition, verify_circuit
from task import task_from_dict


@dataclass
class SolvedExample:
    """A verified solution: task + explanation + circuit."""
    task: GraphStateTask
    explanation: str
    circuit: CircuitDefinition
    fidelity: float = 1.0

    def to_dict(self) -> dict:
        return {
            "task": self.task.to_dict(),
            "explanation": self.explanation,
            "circuit": self.circuit.to_dict(),
            "fidelity": self.fidelity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SolvedExample":
        return cls(
            task=task_from_dict(d["task"]),
            explanation=d["explanation"],
            circuit=CircuitDefinition.from_dict(d["circuit"]),
            fidelity=d.get("fidelity", 1.0),
        )

    def format_for_prompt(self) -> str:
        """Format this example for inclusion in an LLM prompt."""
        lines = [
            f"Target: {self.task.description()}",
            f"Strategy: {self.explanation}",
            f"Circuit: {self.circuit.to_json()}",
        ]
        return "\n".join(lines)


class SolutionLibrary:
    """Growing library of verified solutions with similarity-based retrieval."""

    def __init__(self):
        self.examples: list[SolvedExample] = []

    def add(self, example: SolvedExample):
        self.examples.append(example)

    def size(self) -> int:
        return len(self.examples)

    def retrieve(
        self,
        task: GraphStateTask,
        k: int = 3,
        rng: random.Random = None,
    ) -> list[SolvedExample]:
        """Retrieve k similar solved examples. Top k-1 by similarity + 1 random."""
        if not self.examples:
            return []
        if len(self.examples) <= k:
            return list(self.examples)

        if rng is None:
            rng = random.Random()

        scored = []
        for ex in self.examples:
            score = self._similarity(task, ex.task)
            scored.append((score, ex))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Top k-1 by similarity
        result = [ex for _, ex in scored[: k - 1]]

        # 1 random from the rest for exploration
        remaining = [ex for _, ex in scored[k - 1 :]]
        if remaining:
            result.append(rng.choice(remaining))
        else:
            result.append(scored[-1][1])

        return result

    @staticmethod
    def _similarity(query, candidate) -> float:
        """Score structural similarity between two tasks.

        Graph-state tasks use the original edge-count + degree-sequence metric.
        Other task kinds fall back to a generic structural-feature distance.
        """
        if query.kind() == "graph_state" and candidate.kind() == "graph_state":
            score = 0.0
            # Same qubit count is most important
            if query.num_qubits == candidate.num_qubits:
                score += 2.0
            # Edge count difference
            score -= 0.3 * abs(len(query.edges) - len(candidate.edges))
            # Degree sequence L1 distance
            dq = list(query.degree_sequence())
            dc = list(candidate.degree_sequence())
            max_len = max(len(dq), len(dc))
            dq += [0] * (max_len - len(dq))
            dc += [0] * (max_len - len(dc))
            l1 = sum(abs(a - b) for a, b in zip(dq, dc))
            score -= 0.5 * l1
            return score

        # Generic structural-feature similarity (e.g. unitary tasks).
        qf = query.structural_features()
        cf = candidate.structural_features()
        score = 0.0
        if qf.get("num_qubits") == cf.get("num_qubits"):
            score += 2.0
        for key in ("edge_count", "generator_gate_count", "t_count"):
            if key in qf and key in cf:
                score -= 0.3 * abs(qf[key] - cf[key])
        return score

    def to_json(self) -> str:
        return json.dumps([ex.to_dict() for ex in self.examples], indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SolutionLibrary":
        lib = cls()
        for d in json.loads(json_str):
            lib.add(SolvedExample.from_dict(d))
        return lib

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "SolutionLibrary":
        with open(path) as f:
            return cls.from_json(f.read())


def create_starter_library() -> SolutionLibrary:
    """Create the initial library of 3 verified examples."""
    lib = SolutionLibrary()

    examples = [
        SolvedExample(
            task=GraphStateTask(
                num_qubits=4,
                edges=[(0, 1), (1, 2), (2, 3)],
                task_id="starter_4q_line",
            ),
            explanation=(
                "Apply H to all 4 qubits to create superpositions, "
                "then apply CZ between each adjacent pair along the line: "
                "0-1, 1-2, 2-3."
            ),
            circuit=CircuitDefinition(
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
            ),
        ),
        SolvedExample(
            task=GraphStateTask(
                num_qubits=4,
                edges=[(0, 1), (0, 2), (0, 3)],
                task_id="starter_4q_star",
            ),
            explanation=(
                "Apply H to all 4 qubits, then apply CZ from the center "
                "qubit 0 to each leaf qubit: 0-1, 0-2, 0-3."
            ),
            circuit=CircuitDefinition(
                num_qubits=4,
                gates=[
                    {"gate": "H", "qubits": [0]},
                    {"gate": "H", "qubits": [1]},
                    {"gate": "H", "qubits": [2]},
                    {"gate": "H", "qubits": [3]},
                    {"gate": "CZ", "qubits": [0, 1]},
                    {"gate": "CZ", "qubits": [0, 2]},
                    {"gate": "CZ", "qubits": [0, 3]},
                ],
            ),
        ),
        SolvedExample(
            task=GraphStateTask(
                num_qubits=5,
                edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
                task_id="starter_5q_cycle",
            ),
            explanation=(
                "Apply H to all 5 qubits, then apply CZ along each edge "
                "of the cycle: 0-1, 1-2, 2-3, 3-4, 4-0."
            ),
            circuit=CircuitDefinition(
                num_qubits=5,
                gates=[
                    {"gate": "H", "qubits": [0]},
                    {"gate": "H", "qubits": [1]},
                    {"gate": "H", "qubits": [2]},
                    {"gate": "H", "qubits": [3]},
                    {"gate": "H", "qubits": [4]},
                    {"gate": "CZ", "qubits": [0, 1]},
                    {"gate": "CZ", "qubits": [1, 2]},
                    {"gate": "CZ", "qubits": [2, 3]},
                    {"gate": "CZ", "qubits": [3, 4]},
                    {"gate": "CZ", "qubits": [4, 0]},
                ],
            ),
        ),
    ]

    # Verify all starter examples
    for ex in examples:
        passed, fidelity = verify_circuit(ex.task, ex.circuit)
        if not passed:
            raise RuntimeError(
                f"Starter example '{ex.task.task_id}' failed verification! "
                f"Fidelity: {fidelity:.6f}"
            )
        ex.fidelity = fidelity
        lib.add(ex)

    return lib
