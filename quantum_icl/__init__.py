"""Quantum-ICL: simulator-verified, self-improving in-context learning for
quantum circuit synthesis.

A frozen LLM proposes circuits as strict JSON; a simulator verifies them
exactly; verified solutions accumulate in a retrieval-indexed library that is
fed back into future prompts. No model fine-tuning.
"""

__all__ = [
    "schema",
    "simulate",
    "tasks",
    "verify",
    "library",
    "retrieval",
    "prompts",
    "llm",
    "metrics",
    "experiment",
    "plots",
]
