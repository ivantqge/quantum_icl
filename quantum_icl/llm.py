"""LLM backends: mock (free), OpenRouter / xAI Grok (API), and local HF (GPU).

All backends return an LLMResponse and expose cumulative token/cost counters.
No backend fine-tunes; they only generate completions.
"""

from dataclasses import dataclass
import json
import os
import random
import time


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class BaseLLM:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.total_cost_usd = 0.0

    def _track(self, resp: LLMResponse):
        self.total_prompt_tokens += resp.prompt_tokens
        self.total_completion_tokens += resp.completion_tokens
        self.total_calls += 1
        self.total_cost_usd += resp.cost_usd
        return resp

    def generate(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError


# --- mock ------------------------------------------------------------------

class MockLLM(BaseLLM):
    """Looks up each task's hidden generator (a correct solution) by matching
    its description in the prompt. Returns it with probability `success_rate`,
    otherwise a perturbed (usually wrong) circuit. Free; for pipeline testing."""

    def __init__(self, tasks, rng=None, success_rate=0.7):
        super().__init__()
        self.rng = rng or random.Random()
        self.success_rate = success_rate
        self._answers = {t.description.strip(): t.generator for t in tasks
                         if t.generator is not None}

    def generate(self, system: str, user: str) -> LLMResponse:
        target = user.split("--- New target ---")[-1]
        gen = None
        for desc, circ in self._answers.items():
            if desc in target:
                gen = circ
                break
        if gen is None:
            return self._track(LLMResponse('{"error": "unmatched task"}'))

        gates = list(gen["gates"])
        if self.rng.random() >= self.success_rate and gates:
            if self.rng.random() < 0.5:
                gates.pop(self.rng.randint(0, len(gates) - 1))
            else:
                q = self.rng.randint(0, gen["num_qubits"] - 1)
                gates.append({"gate": "H", "qubits": [q]})
        circ = {"num_qubits": gen["num_qubits"], "gates": gates}
        text = f"Strategy: apply the generating gates.\n```json\n{json.dumps(circ)}\n```"
        return self._track(LLMResponse(text))


# --- OpenAI-compatible APIs (OpenRouter, xAI Grok) -------------------------

GROK_PRICING = {  # USD per 1K tokens (input, output)
    "grok-3-mini": (0.0003, 0.0005), "grok-3": (0.003, 0.015),
    "grok-4": (0.003, 0.015), "grok-4-fast": (0.0002, 0.0005),
}


class _OpenAICompatibleLLM(BaseLLM):
    API_NAME = "openai-compatible"
    BASE_URL = ""
    API_KEY_ENV = ""
    PRICING = {}
    DEFAULT_HEADERS = {}
    REQUEST_USAGE_ACCOUNTING = False

    def __init__(self, model, max_tokens=1024, temperature=0.0,
                 max_api_retries=5, timeout=60.0):
        super().__init__()
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")
        key = os.environ.get(self.API_KEY_ENV)
        if not key:
            raise ValueError(f"Set the {self.API_KEY_ENV} environment variable")
        self._openai = openai
        self.client = openai.OpenAI(
            api_key=key, base_url=self.BASE_URL, timeout=timeout,
            default_headers=self.DEFAULT_HEADERS or None,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_api_retries = max_api_retries

    def generate(self, system: str, user: str) -> LLMResponse:
        openai = self._openai
        retryable = (openai.RateLimitError, openai.APIConnectionError,
                     openai.APITimeoutError)
        kwargs = dict(
            model=self.model, max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        if self.REQUEST_USAGE_ACCOUNTING:
            kwargs["extra_body"] = {"usage": {"include": True}}
        last = None
        for attempt in range(self.max_api_retries):
            try:
                r = self.client.chat.completions.create(**kwargs)
                return self._track(self._to_response(r))
            except retryable as e:
                last = e
                wait = 2 ** attempt
                print(f"    {self.API_NAME}: retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"{self.API_NAME} failed after retries: {last}")

    def _to_response(self, r) -> LLMResponse:
        text = r.choices[0].message.content or ""
        usage = getattr(r, "usage", None)
        if usage is None:
            return LLMResponse(text)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        cost = getattr(usage, "cost", None)
        if cost is None:
            a, b = self.PRICING.get(self.model, (0.0, 0.0))
            cost = pt / 1000.0 * a + ct / 1000.0 * b
        return LLMResponse(text, pt, ct, float(cost))


class OpenRouterLLM(_OpenAICompatibleLLM):
    API_NAME = "OpenRouter"
    BASE_URL = "https://openrouter.ai/api/v1"
    API_KEY_ENV = "OPENROUTER_API_KEY"
    DEFAULT_HEADERS = {"X-Title": "quantum-icl"}
    REQUEST_USAGE_ACCOUNTING = True

    def __init__(self, model="openai/gpt-4o-mini", **kw):
        super().__init__(model=model, **kw)


class GrokLLM(_OpenAICompatibleLLM):
    API_NAME = "xAI Grok"
    BASE_URL = "https://api.x.ai/v1"
    API_KEY_ENV = "XAI_API_KEY"
    PRICING = GROK_PRICING

    def __init__(self, model="grok-3-mini", **kw):
        super().__init__(model=model, **kw)


# --- local HF model (GPU) --------------------------------------------------

class LocalHFLLM(BaseLLM):
    """Local Hugging Face causal LM (runs on GPU). Free to run.

    Loads `model` with transformers, applies its chat template, and greedily
    decodes. Intended for the bulk experiments on a GPU node; the API backends
    are reserved for cheap smoke/pilot runs.
    """

    def __init__(self, model, max_tokens=1024, temperature=0.0, device="auto",
                 adapter_path=None):
        super().__init__()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype="auto", device_map=device,
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, system: str, user: str) -> LLMResponse:
        torch = self._torch
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_tokens = int(enc["input_ids"].shape[-1])
        gen_kwargs = dict(
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        gen_ids = out[0][prompt_tokens:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return self._track(LLMResponse(text, prompt_tokens, int(gen_ids.shape[-1])))


def make_llm(backend: str, tasks=None, **kwargs) -> BaseLLM:
    """Factory. `tasks` is required for the mock backend (answer lookup)."""
    backend = backend.lower()
    if backend == "mock":
        return MockLLM(tasks or [], rng=kwargs.get("rng"),
                       success_rate=kwargs.get("success_rate", 0.7))
    if backend == "openrouter":
        return OpenRouterLLM(model=kwargs.get("model", "openai/gpt-4o-mini"),
                             temperature=kwargs.get("temperature", 0.0),
                             max_tokens=kwargs.get("max_tokens", 1024))
    if backend == "grok":
        return GrokLLM(model=kwargs.get("model", "grok-3-mini"),
                       temperature=kwargs.get("temperature", 0.0),
                       max_tokens=kwargs.get("max_tokens", 1024))
    if backend == "local":
        return LocalHFLLM(model=kwargs["model"],
                          temperature=kwargs.get("temperature", 0.0),
                          max_tokens=kwargs.get("max_tokens", 1024),
                          adapter_path=kwargs.get("adapter_path"))
    raise ValueError(f"unknown backend {backend!r}")
