"""Circuit JSON schema: validation and robust extraction from LLM text.

Schema:
    {
      "num_qubits": int,
      "gates": [{"gate": <name>, "qubits": [int, ...]}, ...]
    }

Supported gate names and arities are fixed here; individual task tiers may
allow only a subset (enforced in verify.py).
"""

import json
import re

# gate name -> number of qubits it acts on
GATE_ARITY = {
    "H": 1, "S": 1, "T": 1, "X": 1, "Y": 1, "Z": 1,
    "CX": 2, "CZ": 2,
}
SUPPORTED_GATES = set(GATE_ARITY)


class SchemaError(ValueError):
    """Raised when a circuit dict violates the schema."""


def extract_json(text: str) -> dict:
    """Extract a circuit JSON object from arbitrary LLM response text.

    Handles markdown code fences, trailing commas, and surrounding prose.
    Raises SchemaError if no parseable object containing "gates" is found.
    """
    if isinstance(text, dict):
        return text

    candidate = None
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        # First balanced {...} block that mentions "gates".
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    block = text[start:i + 1]
                    if "gates" in block:
                        candidate = block
                        break
    if candidate is None:
        raise SchemaError("No JSON circuit object found in response")

    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)       # trailing commas
    candidate = re.sub(r"//.*$", "", candidate, flags=re.MULTILINE)  # comments
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise SchemaError(f"Invalid JSON: {e}")


def validate_circuit_dict(data: dict, allowed_gates=None) -> None:
    """Validate a circuit dict against the schema.

    If `allowed_gates` is given, gate names are additionally restricted to that
    set (used to enforce a task tier's gate set). Raises SchemaError on any
    violation.
    """
    if not isinstance(data, dict):
        raise SchemaError("Circuit must be a JSON object")
    if "num_qubits" not in data or "gates" not in data:
        raise SchemaError("Circuit missing 'num_qubits' and/or 'gates'")

    n = data["num_qubits"]
    if not isinstance(n, int) or n <= 0:
        raise SchemaError(f"num_qubits must be a positive int, got {n!r}")
    if not isinstance(data["gates"], list):
        raise SchemaError("'gates' must be a list")

    allowed = set(allowed_gates) if allowed_gates is not None else SUPPORTED_GATES

    for idx, g in enumerate(data["gates"]):
        if not isinstance(g, dict) or "gate" not in g or "qubits" not in g:
            raise SchemaError(f"gate {idx}: must have 'gate' and 'qubits'")
        name = g["gate"]
        qubits = g["qubits"]
        if name not in SUPPORTED_GATES:
            raise SchemaError(f"gate {idx}: unsupported gate {name!r}")
        if name not in allowed:
            raise SchemaError(
                f"gate {idx}: gate {name!r} not allowed for this task "
                f"(allowed: {sorted(allowed)})"
            )
        if not isinstance(qubits, list) or len(qubits) != GATE_ARITY[name]:
            raise SchemaError(
                f"gate {idx}: {name} needs {GATE_ARITY[name]} qubit(s), "
                f"got {qubits!r}"
            )
        for q in qubits:
            if not isinstance(q, int) or not (0 <= q < n):
                raise SchemaError(
                    f"gate {idx}: qubit index {q!r} out of range [0, {n})"
                )
        if len(set(qubits)) != len(qubits):
            raise SchemaError(f"gate {idx}: repeated qubit in {qubits!r}")
