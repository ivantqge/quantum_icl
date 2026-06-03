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


_TIER_HINTS = {
    "B": (
        "\n\nThis is a stabilizer-state preparation task. Build superpositions "
        "with H, fix relative phases with S, and create the correct "
        "entanglement/correlations with CX/CZ."
    ),
    "C": (
        "\n\nThis is a Clifford unitary-synthesis task: build the exact unitary "
        "from H, S, CX. Reason column by column about how basis states map."
    ),
    "D": (
        "\n\nThis is a Clifford+T task: the target generally needs T gates. "
        "Decompose into Clifford layers plus T rotations; reason about how each "
        "computational basis state should transform."
    ),
}


_COT_SUFFIX = (
    "\n\nThink step by step BEFORE emitting JSON:\n"
    "  1. Identify the structure of the target (sparsity, dominant entries, "
    "any obvious factorization).\n"
    "  2. Pick a gate sequence that you believe realizes the target, "
    "justifying each gate's role in 1-2 sentences.\n"
    "  3. Mentally execute the sequence on |0...0> and check it matches the "
    "target up to global phase.\n"
    "  4. ONLY after this reasoning, emit the strict JSON fenced block.\n"
    "Keep the reasoning concise (max ~120 words) but explicit."
)


def system_prompt(task, prompt_variant: str = "default") -> str:
    allowed = ", ".join(f"{g} ({_GATE_HELP[g]})" for g in task.gate_set)
    base = (
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
        + _TIER_HINTS.get(task.tier, "")
    )
    if prompt_variant == "cot":
        base += _COT_SUFFIX
    return base


def format_example(ex: dict) -> str:
    return (
        f"Target:\n{ex['description']}\n"
        f"Solution circuit:\n```json\n{json.dumps(ex['solution_circuit'])}\n```"
    )


def build_messages(task, examples, feedback=None, prompt_variant="default") -> tuple:
    """Return (system_message, user_message). `examples` is a list of entries
    each having 'description' and 'solution_circuit' (library entries or fixed
    examples); pass [] for zero-shot. `feedback`, if given, describes the
    previous failed attempt so the model can refine (self-refinement loop)."""
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

    if feedback:
        parts.append("--- Your previous attempt was INCORRECT ---")
        if feedback.get("prev_circuit_json"):
            parts.append("Circuit you proposed:")
            parts.append("```json")
            parts.append(feedback["prev_circuit_json"])
            parts.append("```")
        if not feedback.get("valid", True):
            parts.append(f"The verifier rejected it: {feedback.get('error', 'invalid')}")
        else:
            parts.append(
                f"It ran but matched the target with fidelity only "
                f"{feedback.get('fidelity', 0.0):.4f} (need > 0.999, up to "
                f"global phase). It is close but not equivalent."
            )
            if feedback.get("produced_state_str"):
                parts.append(
                    f"Your circuit produced this state instead:\n  "
                    f"{feedback['produced_state_str']}"
                )
                parts.append(
                    "Compare it entry-by-entry with the target state above and "
                    "change the gates to remove the difference."
                )
        parts.append("Diagnose what is wrong and output a corrected circuit.")
        parts.append("")

    parts.append(
        "Give a one-sentence strategy, then output the circuit as JSON in a "
        "```json fenced block matching the schema exactly."
    )
    return system_prompt(task, prompt_variant=prompt_variant), "\n".join(parts)


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
