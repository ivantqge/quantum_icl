"""Dump the exact prompts used for every tier/variant/condition.

Writes to paper/RAW_PROMPTS.md so we have a reproducible record of what
the LLM actually saw.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_icl import tasks as T
from quantum_icl.prompts import build_messages, system_prompt, fixed_examples


OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "RAW_PROMPTS.md")


def fmt_block(label, text):
    return f"\n### {label}\n\n```text\n{text}\n```\n"


def sample_task(tier, seed=42):
    return T.generate_tasks(tier, 1, rng=random.Random(seed))[0]


def main():
    parts = ["# Raw prompts used in Quantum-ICL\n\n"
             "This document shows the exact prompts sent to the LLM for every "
             "tier and prompt variant. Reproducible from any commit via "
             "`python scripts/dump_prompts.py`.\n"]

    # ----- System prompts per tier -----
    parts.append("\n## 1. System prompts (per tier × prompt variant)\n")
    for tier in ["A", "B", "C_lite", "D_lite", "D_mid"]:
        task = sample_task(tier)
        parts.append(f"\n### Tier `{tier}` — default prompt\n")
        parts.append(f"\n```text\n{system_prompt(task, prompt_variant='default')}\n```\n")
        parts.append(f"\n### Tier `{tier}` — `cot` variant (adds CoT suffix)\n")
        parts.append(f"\n```text\n{system_prompt(task, prompt_variant='cot')}\n```\n")

    # ----- User prompts (target descriptions) -----
    parts.append("\n## 2. User prompts — target descriptions per tier\n")
    parts.append("These are inserted into the user message after the "
                 "`--- New target ---` marker.\n")
    for tier in ["A", "B", "C_lite", "D_lite", "D_mid"]:
        task = sample_task(tier)
        parts.append(f"\n### Tier `{tier}` — sample target (seed 42)\n")
        parts.append(f"\n```text\n{task.description}\n```\n")

    # ----- Full assembled user prompt (zero_shot) -----
    parts.append("\n## 3. Full assembled prompts (system + user)\n")
    parts.append("Showing the **entire** message stack the LLM receives, by "
                 "condition.\n")

    for tier in ["B", "D_lite"]:
        task = sample_task(tier, seed=100)
        # zero_shot
        sys_msg, user_msg = build_messages(task, examples=[], feedback=None)
        parts.append(f"\n### Tier `{tier}` — `zero_shot`\n")
        parts.append("\n**System message:**\n")
        parts.append(f"\n```text\n{sys_msg}\n```\n")
        parts.append("\n**User message:**\n")
        parts.append(f"\n```text\n{user_msg}\n```\n")

        # with feedback (fake feedback for illustration)
        fake_feedback = {
            "prev_circuit_json": '{"num_qubits": 2, "gates": [{"gate": "H", "qubits": [0]}]}',
            "fidelity": 0.7071,
            "valid": True,
            "error": "",
            "produced_state_str": "[+0.7071+0.0000j, +0.0000+0.0000j, +0.7071+0.0000j, +0.0000+0.0000j]",
        }
        sys_msg, user_msg = build_messages(task, examples=[], feedback=fake_feedback)
        parts.append(f"\n### Tier `{tier}` — `feedback_only` after one failed attempt\n")
        parts.append(f"\n```text\n{user_msg}\n```\n")

        # structural retrieval + feedback
        ex = fixed_examples(tier, k=2)
        sys_msg, user_msg = build_messages(task, examples=ex, feedback=fake_feedback)
        parts.append(f"\n### Tier `{tier}` — `structural_retrieval_plus_feedback` (2 examples + feedback)\n")
        parts.append(f"\n```text\n{user_msg}\n```\n")

        # CoT system prompt + retrieval + feedback
        sys_msg_cot, user_msg = build_messages(task, examples=ex, feedback=fake_feedback,
                                               prompt_variant="cot")
        parts.append(f"\n### Tier `{tier}` — `structural_retrieval_plus_feedback` + CoT system prompt\n")
        parts.append("\n**System message (with CoT suffix):**\n")
        parts.append(f"\n```text\n{sys_msg_cot}\n```\n")

    # ----- Notes -----
    parts.append("\n## 4. Notes\n")
    parts.append("""
- All prompts use **strict-JSON** instruction with a markdown-fenced
  `\\`\\`\\`json ... \\`\\`\\`` block. Parsing is done by the verifier in
  `quantum_icl/schema.py::extract_json`, which is robust to surrounding prose,
  trailing commas, and `//` line comments.
- The **feedback augmentation** is inserted *between* `--- New target ---` and
  the final "Give a one-sentence strategy..." line, so the model always sees
  the target last (recency-biased).
- For state tiers (A, B), feedback includes the *actual state vector* the
  proposed circuit produced, so the model can do entry-by-entry comparison.
  For unitary tiers (C_lite, D_lite, D_mid), only the fidelity scalar is fed
  back (the unitary would be too large to render usefully).
- Examples are drawn from a per-tier library of `SolvedExample` objects.
  In retrieval conditions, the library *grows online* as tasks are verified
  in the same run (same seeds, same order). Examples are formatted with
  `format_example()` which renders each as `Target:\\n{description}\\nSolution
  circuit:\\n\\`\\`\\`json {circuit}\\`\\`\\``.
- Temperature is 0 for all default runs; the Best-of-N variant uses
  temperature 0.4--0.5 with `attempts=5` and `use_feedback=False` so the
  attempts are independent samples.
""")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {OUT}")
    print(f"size: {os.path.getsize(OUT) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
