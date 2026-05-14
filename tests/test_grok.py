"""Tests for the GrokLLM backend (no network calls)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm as llm_mod
from llm import GrokLLM


class _FakeUsage:
    prompt_tokens = 100
    completion_tokens = 40
    total_tokens = 140


class _FakeMessage:
    content = "strategy text\n```json\n{\"num_qubits\": 1, \"gates\": []}\n```"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]
    usage = _FakeUsage()


class _FakeCompletions:
    def create(self, **kwargs):
        return _FakeResponse()


class _FakeChat:
    completions = _FakeCompletions()


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.chat = _FakeChat()


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="XAI_API_KEY"):
        GrokLLM()


def test_generate_returns_string_and_records_usage(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    import openai
    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    grok = GrokLLM(model="grok-3-mini")
    out = grok.generate("system prompt", "user prompt")

    assert isinstance(out, str)
    assert "gates" in out

    usage = grok.get_last_usage()
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 40
    assert usage["est_cost_usd"] > 0

    assert grok.total_calls == 1
    assert grok.total_prompt_tokens == 100
    assert grok.total_completion_tokens == 40
    assert grok.total_cost_usd > 0


def test_unknown_model_cost_is_zero(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    import openai
    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    grok = GrokLLM(model="grok-does-not-exist")
    grok.generate("s", "u")
    assert grok.get_last_usage()["est_cost_usd"] == 0.0
