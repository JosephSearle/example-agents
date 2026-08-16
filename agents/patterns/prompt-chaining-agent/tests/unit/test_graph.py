"""Unit tests for prompt_chaining_agent.graph — pure logic and a stubbed model, no network."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
from prompt_chaining_agent.graph import build_chain, gate_check_outline, invoke_config
import pytest

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import InMemorySaver

pytestmark = pytest.mark.unit

_STEP_PROMPTS = {
    "outline": "Write an outline.",
    "draft": "Write a draft.",
    "polish": "Polish the draft.",
}


def test_gate_check_outline_passes_on_well_formed_outline() -> None:
    gate_check_outline("Intro\nBody\nConclusion")


def test_gate_check_outline_raises_on_too_thin_outline() -> None:
    with pytest.raises(ValueError, match="only has 1 section"):
        gate_check_outline("Just one line")


def test_invoke_config_sets_thread_id() -> None:
    config = invoke_config("some-thread-id")

    assert config["configurable"]["thread_id"] == "some-thread-id"


def test_chain_runs_outline_draft_polish_in_order(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(
        [
            AIMessage(content="Intro\nBody\nConclusion"),
            AIMessage(content="a full draft"),
            AIMessage(content="a polished draft"),
        ]
    )

    class _FakeModel:
        def invoke(self, _prompt: str) -> AIMessage:
            return next(responses)

    monkeypatch.setattr("prompt_chaining_agent.graph.get_chat_model", lambda _route: _FakeModel())

    chain = build_chain(checkpointer=in_memory_checkpointer, step_prompts=_STEP_PROMPTS)
    config = invoke_config("thread-1")
    result = chain.invoke(
        {"topic": "some topic", "outline": "", "draft": "", "final": ""}, config=config
    )

    assert result["outline"] == "Intro\nBody\nConclusion"
    assert result["draft"] == "a full draft"
    assert result["final"] == "a polished draft"


def test_chain_stops_when_gate_check_fails(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeModel:
        def invoke(self, _prompt: str) -> AIMessage:
            return AIMessage(content="one line only")

    monkeypatch.setattr("prompt_chaining_agent.graph.get_chat_model", lambda _route: _FakeModel())

    chain = build_chain(checkpointer=in_memory_checkpointer, step_prompts=_STEP_PROMPTS)
    config = invoke_config("thread-2")

    with pytest.raises(ValueError, match="expected at least"):
        chain.invoke(
            {"topic": "some topic", "outline": "", "draft": "", "final": ""}, config=config
        )
