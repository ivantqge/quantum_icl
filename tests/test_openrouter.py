"""Tests for the OpenRouterLLM backend (no network calls)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import OpenRouterLLM


class _FakeUsage:
    prompt_tokens = 120
    completion_tokens = 30
    total_tokens = 150
    cost = 0.000345  # OpenRouter reports per-call cost via usage accounting


class _FakeMessage:
    content = "strategy\n```json\n{\"num_qubits\": 1, \"gates\": []}\n```"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]
    usage = _FakeUsage()


class _FakeCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse()


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.chat = _FakeChat()


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterLLM()


def test_generate_uses_provider_cost_and_usage_accounting(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import openai
    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    llm = OpenRouterLLM(model="openai/gpt-4o-mini")
    out = llm.generate("system", "user")
    assert "gates" in out

    usage = llm.get_last_usage()
    assert usage["prompt_tokens"] == 120
    assert usage["completion_tokens"] == 30
    # Cost comes straight from the provider, not a local pricing table.
    assert usage["est_cost_usd"] == pytest.approx(0.000345)

    # Usage accounting must have been requested in the API call.
    assert llm.client.chat.completions.last_kwargs["extra_body"] == {
        "usage": {"include": True}
    }
