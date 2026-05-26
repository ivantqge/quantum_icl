"""Prompt construction for each experimental condition.

build_messages() is condition-agnostic: callers pass the in-context examples
(none, fixed, or retrieved) and the prompt is assembled identically, so the
only thing that varies across conditions is which examples are supplied.
"""

import json
import random

from .tasks import generate_tasks

_GATE_HELP = {
    "H": "Hadamard", "S": "phase (pi/2)", "T": "pi/4 phase",
    "X": "Pauli-X", "Y": "Pauli-Y", "Z": "Pauli-Z",
    "CX": "CNOT [control, target]", "CZ": "controlled-Z",
}


def system_prompt(task) -> str:
    allowed = ", ".join(f"{g} ({_GATE_HELP[g]})" for g in task.gate_set)
    return (
        "You are a quantum circuit synthesizer. Given a target, output a "
        "circuit that realizes it, starting from |0...0>.\n\n"
        f"Allowed gates for THIS task: {allowed}.\n"
        "Single-qubit gates take one qubit index; CX and CZ take two "
        "(CX is [control, target]).\n\n"
        "Output a strict JSON object of the form:\n"
        '{"num_qubits": N, "gates": [{"gate": "H", "qubits": [0]}, '
        '{"gate": "CX", "qubits": [0, 1]}]}\n\n'
        "Gates are applied left to right. Your circuit is correct if it "
        "reproduces the target up to a global phase. Use only the allowed "
        "gates and qubit indices in [0, N)."
    )


def format_example(ex: dict) -> str:
    return (
        f"Target:\n{ex['description']}\n"
        f"Solution circuit:\n```json\n{json.dumps(ex['solution_circuit'])}\n```"
    )


def build_messages(task, examples) -> tuple:
    """Return (system_message, user_message). `examples` is a list of entries
    each having 'description' and 'solution_circuit' (library entries or fixed
    examples); pass [] for zero-shot."""
    parts = []
    if examples:
        parts.append("Here are solved examples:\n")
        for i, ex in enumerate(examples, 1):
            parts.append(f"--- Example {i} ---")
            parts.append(format_example(ex))
            parts.append("")
    parts.append("--- New target ---")
    parts.append(task.description)
    parts.append("")
    parts.append(
        "Give a one-sentence strategy, then output the circuit as JSON in a "
        "```json fenced block matching the schema exactly."
    )
    return system_prompt(task), "\n".join(parts)


# --- fixed few-shot examples ----------------------------------------------

_FIXED_CACHE = {}


def fixed_examples(tier: str, k: int = 2, seed: int = 9999) -> list:
    """Hand-fixed few-shot examples for a tier.

    Built once per tier from generator circuits (which are correct solutions
    by construction) using a dedicated seed disjoint from evaluation tasks.
    """
    key = (tier, k, seed)
    if key in _FIXED_CACHE:
        return _FIXED_CACHE[key]
    rng = random.Random(seed)
    tasks = generate_tasks(tier, k, rng=rng)
    examples = [
        {"description": t.description, "solution_circuit": t.generator}
        for t in tasks
    ]
    _FIXED_CACHE[key] = examples
    return examples
