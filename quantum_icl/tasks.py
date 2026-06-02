"""Task generation for the four tiers.

Tier A: graph-state preparation        (H, CZ;            verify state fidelity)
Tier B: stabilizer-state synthesis     (H, S, CX, CZ;     verify state fidelity)
Tier C: Clifford unitary synthesis     (H, S, CX;         verify process fidelity)
Tier D: Clifford+T hidden-circuit synth(H, S, T, CX, CZ;  verify process fidelity)

Targets for B/C/D are produced by a hidden random "generator" circuit; the
generator is never shown to the LLM but its statistics are recorded as task
features (and used by the mock backend).
"""

from dataclasses import dataclass, field
import os
import random

import numpy as np

from .schema import GATE_ARITY
from .simulate import (
    state_vector, circuit_unitary, circuit_metrics, stabilizer_generators,
    state_features, unitary_features,
)

TIER_GATE_SETS = {
    "A": ["H", "CZ"],
    "B": ["H", "S", "CX", "CZ"],
    "C": ["H", "S", "CX"],
    "D": ["H", "S", "T", "CX", "CZ"],
    # "lite" tiers: small/shallow targets to keep difficulty in the 20-70%
    # baseline-success regime where ICL/feedback effects are visible.
    "C_lite": ["H", "S", "X", "Y", "Z"],         # 1-qubit Clifford (24 elements)
    "D_lite": ["H", "S", "T", "X", "Y", "Z"],    # 1-qubit Clifford+T (low T-count)
    "D_mid":  ["H", "S", "T", "CX", "CZ"],       # 2-qubit Clifford+T (depth ~6-10)
}
TIER_TARGET_KIND = {
    "A": "state", "B": "state",
    "C": "unitary", "D": "unitary",
    "C_lite": "unitary", "D_lite": "unitary", "D_mid": "unitary",
}


@dataclass
class Task:
    task_id: str
    tier: str
    num_qubits: int
    target_kind: str            # "state" | "unitary"
    target: np.ndarray          # state vector or unitary matrix
    gate_set: list              # allowed gate names
    features: dict              # TARGET-COMPUTABLE features (used by retrieval)
    description: str
    generator: dict = None      # hidden generator circuit dict (kept for analysis)
    # Oracle features derived from the hidden generator (gate count, T-count,
    # depth, ...). NEVER used by default retrieval; opt-in for diagnostics only.
    oracle_features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        t = np.asarray(self.target)
        return {
            "task_id": self.task_id,
            "tier": self.tier,
            "num_qubits": self.num_qubits,
            "target_kind": self.target_kind,
            "target": {"real": t.real.tolist(), "imag": t.imag.tolist()},
            "gate_set": list(self.gate_set),
            "features": self.features,
            "description": self.description,
            "generator": self.generator,
            "oracle_features": self.oracle_features,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        t = d["target"]
        target = np.array(t["real"], dtype=complex) + 1j * np.array(t["imag"])
        return cls(
            task_id=d["task_id"], tier=d["tier"], num_qubits=d["num_qubits"],
            target_kind=d["target_kind"], target=target,
            gate_set=d["gate_set"], features=d["features"],
            description=d["description"], generator=d.get("generator"),
            oracle_features=d.get("oracle_features", {}),
        )


# --- helpers ---------------------------------------------------------------

def _random_circuit_dict(num_qubits, gate_set, n_gates, rng):
    """Random circuit dict using only gates whose arity fits num_qubits."""
    names = [g for g in gate_set if GATE_ARITY[g] <= num_qubits]
    if not names:
        raise ValueError(f"no gates in {gate_set} fit {num_qubits} qubits")
    gates = []
    for _ in range(n_gates):
        name = rng.choice(names)
        qs = rng.sample(range(num_qubits), GATE_ARITY[name])
        gates.append({"gate": name, "qubits": qs})
    return {"num_qubits": num_qubits, "gates": gates}


def _target_hash(target):
    return abs(hash(np.round(np.asarray(target), 6).tobytes())) % (10 ** 10)


def _fmt_complex(z, p=4):
    return f"{z.real:+.{p}f}{z.imag:+.{p}f}j"


def _describe_state(vec):
    amps = ", ".join(_fmt_complex(z) for z in np.asarray(vec).reshape(-1))
    return ("Target state amplitudes in computational-basis order "
            f"|0...0> ... |1...1>:\n  [{amps}]")


def _describe_unitary(u):
    rows = ["  [" + ", ".join(_fmt_complex(z) for z in row) + "]"
            for row in np.asarray(u)]
    return "Target unitary matrix U (rows = output basis, cols = input basis):\n" + \
           "\n".join(rows)


def _describe_stabilizer(state, n):
    gens = stabilizer_generators(state, n)
    lines = [
        "Target stabilizer state, specified by its stabilizer generators",
        "(Pauli operators that fix the state with the given sign; the leftmost "
        "Pauli letter acts on qubit 0):",
    ]
    lines += [f"  g{i} = {g}" for i, g in enumerate(gens, 1)]
    lines.append(
        "Prepare the common +1 eigenstate of all g_i, starting from |0...0>."
    )
    lines.append(_describe_state(state))
    return "\n".join(lines)


def _connected_components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    return len({find(i) for i in range(n)})


# --- generators ------------------------------------------------------------

def gen_graph_state_tasks(n_tasks, qubit_range=(4, 6), edge_prob=0.5, rng=None):
    rng = rng or random.Random()
    seen, tasks = set(), []
    while len(tasks) < n_tasks:
        n = rng.randint(*qubit_range)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if rng.random() < edge_prob]
        if not edges:
            continue
        key = (n, tuple(sorted(edges)))
        if key in seen:
            continue
        seen.add(key)

        gates = [{"gate": "H", "qubits": [i]} for i in range(n)]
        gates += [{"gate": "CZ", "qubits": [a, b]} for a, b in edges]
        gen = {"num_qubits": n, "gates": gates}
        target = state_vector(gen)

        degrees = [0] * n
        for a, b in edges:
            degrees[a] += 1
            degrees[b] += 1
        features = {
            "num_qubits": n,
            "num_edges": len(edges),
            "num_components": _connected_components(n, edges),
            "degree_sequence": sorted(degrees),
            "gate_set": "|".join(TIER_GATE_SETS["A"]),
        }
        oracle_features = {
            "gen_gate_count": len(gates),
            "gen_two_qubit_gate_count": len(edges),
            "gen_t_count": 0,
            "gen_depth": circuit_metrics(gen)["depth"],
        }
        desc = (f"Graph state on {n} qubits.\nEdges: {edges}\n"
                "Prepare |G> = (prod_{(i,j) in E} CZ_ij) H^{⊗n} |0...0>.")
        tasks.append(Task(
            task_id=f"A_{n}q_{_target_hash(target)}", tier="A", num_qubits=n,
            target_kind="state", target=target, gate_set=TIER_GATE_SETS["A"],
            features=features, description=desc, generator=gen,
            oracle_features=oracle_features,
        ))
    return tasks


def _gen_hidden(tier, n_tasks, qubit_range, gen_gate_range, rng):
    """Shared generator for tiers B/C/D (hidden random circuit)."""
    rng = rng or random.Random()
    gate_set = TIER_GATE_SETS[tier]
    kind = TIER_TARGET_KIND[tier]
    seen, tasks = set(), []
    attempts, cap = 0, n_tasks * 300
    while len(tasks) < n_tasks and attempts < cap:
        attempts += 1
        n = rng.randint(*qubit_range)
        n_gates = rng.randint(*gen_gate_range)
        gen = _random_circuit_dict(n, gate_set, n_gates, rng)
        target = state_vector(gen) if kind == "state" else circuit_unitary(gen)
        key = (n, _target_hash(target))
        if key in seen:
            continue
        seen.add(key)

        m = circuit_metrics(gen)
        # Target-computable features go into Task.features (used by retrieval);
        # hidden-generator stats go into oracle_features (diagnostics only).
        if kind == "state":
            features = state_features(target, n)
        else:
            features = unitary_features(target, n)
        features["gate_set"] = "|".join(gate_set)
        oracle_features = {
            "gen_gate_count": m["gate_count"],
            "gen_two_qubit_gate_count": m["two_qubit_gate_count"],
            "gen_depth": m["depth"],
            "gen_t_count": m["t_count"],
        }
        # Tier B optionally augments amplitudes with stabilizer generators.
        # Empirically, raw amplitudes work better for weaker models, so that is
        # the default; set QICL_STABILIZER_DESC=1 to include generators.
        if tier == "B" and os.environ.get("QICL_STABILIZER_DESC") == "1":
            body = _describe_stabilizer(target, n)
        elif kind == "state":
            body = _describe_state(target)
        else:
            body = _describe_unitary(target)
        desc = f"{tier}-tier synthesis on {n} qubits using gates {gate_set}.\n{body}"
        tasks.append(Task(
            task_id=f"{tier}_{n}q_{_target_hash(target)}", tier=tier,
            num_qubits=n, target_kind=kind, target=target,
            gate_set=gate_set, features=features, description=desc,
            generator=gen, oracle_features=oracle_features,
        ))
    if len(tasks) < n_tasks:
        print(f"  [warn] tier {tier}: dedup ceiling reached "
              f"({len(tasks)}/{n_tasks}); proceeding with what we have.",
              flush=True)
    return tasks


def gen_stabilizer_tasks(n_tasks, qubit_range=(2, 3), gen_gate_range=(3, 8), rng=None):
    return _gen_hidden("B", n_tasks, qubit_range, gen_gate_range, rng)


def gen_clifford_tasks(n_tasks, qubit_range=(2, 2), gen_gate_range=(3, 10), rng=None):
    return _gen_hidden("C", n_tasks, qubit_range, gen_gate_range, rng)


def gen_cliffordT_tasks(n_tasks, qubit_range=(2, 2), gen_gate_range=(4, 12), rng=None):
    return _gen_hidden("D", n_tasks, qubit_range, gen_gate_range, rng)


def gen_clifford_lite_tasks(n_tasks, qubit_range=(1, 1), gen_gate_range=(1, 6), rng=None):
    """1-qubit Clifford unitaries: small enough that examples are highly transferable."""
    return _gen_hidden("C_lite", n_tasks, qubit_range, gen_gate_range, rng)


def gen_cliffordT_lite_tasks(n_tasks, qubit_range=(1, 1), gen_gate_range=(1, 6), rng=None):
    """1-qubit Clifford+T unitaries with low T-count (random {H,S,T,X,Y,Z} sequences)."""
    return _gen_hidden("D_lite", n_tasks, qubit_range, gen_gate_range, rng)


def gen_cliffordT_mid_tasks(n_tasks, qubit_range=(2, 2), gen_gate_range=(6, 10), rng=None):
    """2-qubit Clifford+T unitaries at intermediate difficulty (depth ~6-10).

    Gate set {H,S,T,CX,CZ}; expected T-count ~1.2-2.0 per task with random
    sampling (1/5 of gates are T). Calibrate via gen_gate_range to hit the
    20-60% baseline zero-shot success regime.
    """
    return _gen_hidden("D_mid", n_tasks, qubit_range, gen_gate_range, rng)


GENERATORS = {
    "A": gen_graph_state_tasks,
    "B": gen_stabilizer_tasks,
    "C": gen_clifford_tasks,
    "D": gen_cliffordT_tasks,
    "C_lite": gen_clifford_lite_tasks,
    "D_lite": gen_cliffordT_lite_tasks,
    "D_mid": gen_cliffordT_mid_tasks,
}


def generate_tasks(tier, n_tasks, rng=None, **kwargs):
    """Generate tasks for a tier. Extra kwargs forwarded to the generator."""
    return GENERATORS[tier](n_tasks, rng=rng, **kwargs)
