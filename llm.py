"""LLM interface: abstract base, mock implementation, and prompt construction."""

from abc import ABC, abstractmethod
import json
import random

from graph_task import GraphStateTask
from circuit import CircuitDefinition
from library import SolvedExample


SYSTEM_PROMPT = """\
You are a quantum circuit designer. You synthesize circuits using only H (Hadamard) \
and CZ (controlled-Z) gates to prepare specific quantum states starting from |0...0>.

Allowed gates:
- H: single-qubit Hadamard gate. JSON format: {"gate": "H", "qubits": [i]}
- CZ: two-qubit controlled-Z gate. JSON format: {"gate": "CZ", "qubits": [i, j]}

Your output format:
1. First, briefly explain your construction strategy (2-3 sentences).
2. Then provide the circuit in this exact JSON format:
```json
{"num_qubits": N, "gates": [{"gate": "H", "qubits": [0]}, ...]}
```

A graph state |G> is a quantum state defined by a graph. Your job is to figure out \
how to construct the circuit that prepares it, given the graph structure."""


CLIFFORD_T_SYSTEM_PROMPT = """\
You are a quantum circuit designer. You synthesize circuits using the universal \
gate set {H, S, T, CNOT} to implement a specified target unitary matrix U, \
acting on n qubits initialized to |0...0>.

Allowed gates:
- H: single-qubit Hadamard. JSON: {"gate": "H", "qubits": [i]}
- S: single-qubit phase gate (pi/2). JSON: {"gate": "S", "qubits": [i]}
- T: single-qubit pi/4 phase gate. JSON: {"gate": "T", "qubits": [i]}
- CNOT: controlled-NOT, control i and target j. JSON: {"gate": "CNOT", "qubits": [i, j]}

Your output format:
1. First, briefly explain your construction strategy (2-3 sentences).
2. Then provide the circuit in this exact JSON format:
```json
{"num_qubits": N, "gates": [{"gate": "H", "qubits": [0]}, ...]}
```

Gates are applied in the listed order. Your circuit is correct if its overall \
unitary equals the target U up to a global phase."""


def build_system_prompt(gate_set: str) -> str:
    """Return the system prompt appropriate for a gate set."""
    if gate_set == "graph_state":
        return SYSTEM_PROMPT
    if gate_set == "clifford_t":
        return CLIFFORD_T_SYSTEM_PROMPT
    raise ValueError(f"No system prompt defined for gate set '{gate_set}'")


def build_prompt(
    task,
    examples: list[SolvedExample],
) -> tuple:
    """Build (system_message, user_message) for the LLM.

    Works for any Task (graph_state or unitary); the system prompt is selected
    from the task's gate set. Returns (str, str).
    """
    user_parts = []

    if examples:
        user_parts.append("Here are some solved examples:\n")
        for i, ex in enumerate(examples, 1):
            user_parts.append(f"--- Example {i} ---")
            user_parts.append(ex.format_for_prompt())
            user_parts.append("")

    user_parts.append("--- New Target ---")
    user_parts.append(task.description())
    user_parts.append("")
    user_parts.append(
        "Please explain your strategy briefly, then provide the circuit JSON."
    )

    gate_set = task.gate_set_name() if hasattr(task, "gate_set_name") else "graph_state"
    return (build_system_prompt(gate_set), "\n".join(user_parts))


class BaseLLM(ABC):
    """Abstract LLM interface."""

    def __init__(self):
        # Usage from the most recent generate() call (None if not tracked).
        self.last_usage = None
        # Cumulative counters across the lifetime of this instance.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.total_cost_usd = 0.0

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Generate a response given system and user messages."""
        ...

    def get_last_usage(self) -> dict:
        """Return token/cost usage from the most recent generate() call.

        Returns an empty dict for backends that do not report usage.
        """
        return self.last_usage or {}


class MockLLM(BaseLLM):
    """Template-based mock LLM for pipeline testing.

    Simulates realistic LLM behavior where having examples helps:
    - With examples (static/growing): 80% correct, 10% missing CZ, 10% extra CZ
    - Without examples (independent): 40% correct, 30% missing CZ, 30% extra CZ

    This models the intuition that few-shot examples significantly help an LLM
    learn the graph state construction pattern.
    """

    def __init__(self, rng: random.Random = None):
        super().__init__()
        self.rng = rng or random.Random()

    def generate(self, system: str, user: str) -> str:
        # Parse the target task from the "New Target" section
        task = self._parse_task_from_prompt(user)
        if task is None:
            return '{"error": "could not parse task"}'

        n = task["num_qubits"]
        edges = task["edges"]
        has_examples = task["has_examples"]

        # Success rates depend on whether examples were provided
        if has_examples:
            correct_rate, missing_rate = 0.80, 0.10
        else:
            correct_rate, missing_rate = 0.40, 0.30

        roll = self.rng.random()
        if roll < correct_rate:
            used_edges = list(edges)
            explanation = (
                "Apply H to all qubits to create superpositions, "
                "then apply CZ to each edge in the graph."
            )
        elif roll < correct_rate + missing_rate:
            used_edges = list(edges)
            if len(used_edges) > 1:
                used_edges.pop(self.rng.randint(0, len(used_edges) - 1))
            explanation = (
                "Apply H to all qubits, then apply CZ to connect "
                "the qubits according to the graph edges."
            )
        else:
            used_edges = list(edges)
            all_possible = [
                (i, j) for i in range(n) for j in range(i + 1, n)
            ]
            extra = [e for e in all_possible if e not in edges]
            if extra:
                used_edges.append(self.rng.choice(extra))
            explanation = (
                "Apply H to all qubits to create the uniform superposition, "
                "then entangle pairs using CZ gates."
            )

        gates = [{"gate": "H", "qubits": [i]} for i in range(n)]
        gates += [{"gate": "CZ", "qubits": list(e)} for e in used_edges]

        circuit_json = json.dumps({"num_qubits": n, "gates": gates})

        return f"{explanation}\n\n```json\n{circuit_json}\n```"

    def _parse_task_from_prompt(self, user: str) -> dict:
        """Extract num_qubits and edges from the 'New Target' section."""
        import re

        # Split on "New Target" to isolate the target section
        parts = user.split("--- New Target ---")
        if len(parts) < 2:
            # No target marker; try to parse the whole thing
            target_section = user
            has_examples = False
        else:
            target_section = parts[-1]
            has_examples = "--- Example" in user

        # Find "N qubits" in the target section
        n_match = re.search(r"(\d+)\s+qubits", target_section)
        if not n_match:
            return None
        n = int(n_match.group(1))

        # Find edges in the target section
        edge_match = re.search(r"Edges:\s*\[([^\]]*)\]", target_section)
        if not edge_match:
            return None

        edge_str = edge_match.group(1)
        edges = []
        for pair in re.findall(r"\((\d+),\s*(\d+)\)", edge_str):
            edges.append((int(pair[0]), int(pair[1])))

        return {"num_qubits": n, "edges": edges, "has_examples": has_examples}


class MockUnitaryLLM(BaseLLM):
    """Template-based mock for unitary-synthesis pipeline testing.

    Constructed with the task pool so it can look up each task's ground-truth
    generator circuit (keyed by the task description, which build_prompt embeds
    verbatim in the user message). With probability `success_rate` it returns
    the exact generator circuit; otherwise it perturbs it (drops or adds a
    gate). This exercises parsing, verification, retrieval, and library growth
    without spending API credits.
    """

    def __init__(self, tasks, rng: random.Random = None, success_rate: float = 0.7):
        super().__init__()
        self.rng = rng or random.Random()
        self.success_rate = success_rate
        self._answers = {}
        for t in tasks:
            gen = getattr(t, "generator_circuit", None)
            if gen is not None:
                self._answers[t.description().strip()] = gen

    def generate(self, system: str, user: str) -> str:
        target_section = user.split("--- New Target ---")[-1].strip()
        gen = None
        for desc, circ in self._answers.items():
            if desc in target_section:
                gen = circ
                break
        if gen is None:
            return '{"error": "could not match task"}'

        gates = list(gen.gates)
        if self.rng.random() < self.success_rate:
            explanation = "Apply the gates composing the target unitary in order."
        elif gates and self.rng.random() < 0.5:
            gates.pop(self.rng.randint(0, len(gates) - 1))
            explanation = "Decompose the unitary into its constituent gates."
        else:
            gates.append(
                {"gate": "T", "qubits": [self.rng.randint(0, gen.num_qubits - 1)]}
            )
            explanation = "Build the target unitary gate by gate."

        circuit_json = json.dumps({"num_qubits": gen.num_qubits, "gates": gates})
        return f"{explanation}\n\n```json\n{circuit_json}\n```"


class GeminiLLM(BaseLLM):
    """Google Gemini API-based LLM.

    Requires: pip install google-generativeai
    Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        super().__init__()
        import os
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "The 'google-generativeai' package is required. Install with: "
                "pip install google-generativeai"
            )

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable"
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
        )

    def generate(self, system: str, user: str) -> str:
        import time
        # system instruction is set on the model; just send the user message
        for attempt in range(3):
            try:
                response = self.model.generate_content(user)
                return response.text
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait = 2 ** attempt
                    print(f"    Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("Gemini API failed after 3 retries")


class AnthropicLLM(BaseLLM):
    """Anthropic API-based LLM (requires anthropic package and API key)."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        super().__init__()
        self.model = model
        try:
            import anthropic
            self.client = anthropic.Anthropic()
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required. Install with: "
                "pip install anthropic"
            )

    def generate(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


# Approximate xAI Grok pricing in USD per 1K tokens: (input, output).
# Unknown models fall back to (0.0, 0.0) so cost reporting degrades gracefully.
GROK_PRICING = {
    "grok-3-mini": (0.0003, 0.0005),
    "grok-3": (0.003, 0.015),
    "grok-4": (0.003, 0.015),
    "grok-4-fast": (0.0002, 0.0005),
}


class GrokLLM(BaseLLM):
    """xAI Grok backend via the OpenAI-compatible API.

    Requires: pip install openai
    Set the XAI_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "grok-3-mini",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        max_api_retries: int = 5,
        timeout: float = 60.0,
    ):
        super().__init__()
        import os
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required. Install with: pip install openai"
            )

        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("Set the XAI_API_KEY environment variable")

        self._openai = openai
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=timeout,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_api_retries = max_api_retries

    def generate(self, system: str, user: str) -> str:
        import time

        openai = self._openai
        retryable = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
        )
        last_err = None
        for attempt in range(self.max_api_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                self._record_usage(response)
                return response.choices[0].message.content or ""
            except retryable as e:
                last_err = e
                wait = 2 ** attempt
                print(f"    Rate limited / connection error, retrying in {wait}s...")
                time.sleep(wait)
                continue
        raise RuntimeError(
            f"xAI Grok API failed after {self.max_api_retries} retries: {last_err}"
        )

    def _record_usage(self, response):
        """Capture token usage and estimated cost from an API response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            self.last_usage = {}
            return
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(
            usage, "total_tokens", prompt_tokens + completion_tokens
        )
        in_per_1k, out_per_1k = GROK_PRICING.get(self.model, (0.0, 0.0))
        cost = (
            (prompt_tokens / 1000.0) * in_per_1k
            + (completion_tokens / 1000.0) * out_per_1k
        )
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "est_cost_usd": cost,
        }
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_calls += 1
        self.total_cost_usd += cost
