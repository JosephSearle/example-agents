"""Unit tests for parallelization_agent.graph — pure logic and a stubbed model, no network."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
from parallelization_agent.graph import (
    ActionItems,
    SeverityAssessment,
    build_sectioning_graph,
    build_voting_graph,
    invoke_config,
)
import pytest

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import InMemorySaver

pytestmark = pytest.mark.unit

_SECTION_PROMPTS = {
    "summarize": "Summarize this incident in one paragraph.",
    "assess_severity": "Assess the severity of this incident.",
    "extract_action_items": "Extract follow-up action items from this incident.",
}


def test_invoke_config_sets_thread_id() -> None:
    config = invoke_config("some-thread-id")

    assert config["configurable"]["thread_id"] == "some-thread-id"


class _FakeStructuredModel:
    """Stands in for `model.with_structured_output(...)`."""

    def __init__(self, result: SeverityAssessment | ActionItems) -> None:
        self._result = result

    def invoke(self, _prompt: str) -> SeverityAssessment | ActionItems:
        return self._result


class _FakeModel:
    """Stands in for the chat model returned by `get_chat_model`. Records every prompt it's
    called with, so tests can assert the three sections ran independently."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def with_structured_output(
        self, schema: type[SeverityAssessment] | type[ActionItems]
    ) -> _FakeStructuredModel:
        if schema is SeverityAssessment:
            return _FakeStructuredModel(SeverityAssessment(severity="high"))
        return _FakeStructuredModel(ActionItems(items=["restart the service", "notify on-call"]))

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append(prompt)
        return AIMessage(content="a one-paragraph summary")


def test_sectioning_graph_populates_all_three_sections(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = _FakeModel()
    monkeypatch.setattr("parallelization_agent.graph.get_chat_model", lambda _route: fake_model)

    sectioning = build_sectioning_graph(
        checkpointer=in_memory_checkpointer, section_prompts=_SECTION_PROMPTS
    )
    config = invoke_config("thread-sectioning")
    result = sectioning.invoke(
        {
            "incident_text": "the database fell over",
            "summary": "",
            "severity": "",
            "action_items": [],
            "report": "",
        },
        config=config,
    )

    assert result["summary"] == "a one-paragraph summary"
    assert result["severity"] == "high"
    assert result["action_items"] == ["restart the service", "notify on-call"]
    # Only the unstructured `summarize` call goes through `_FakeModel.invoke` directly — the
    # other two sections call through `with_structured_output`'s stand-in instead.
    assert len(fake_model.calls) == 1


def test_aggregate_report_combines_all_three_sections(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("parallelization_agent.graph.get_chat_model", lambda _route: _FakeModel())

    sectioning = build_sectioning_graph(
        checkpointer=in_memory_checkpointer, section_prompts=_SECTION_PROMPTS
    )
    config = invoke_config("thread-aggregate")
    result = sectioning.invoke(
        {
            "incident_text": "the database fell over",
            "summary": "",
            "severity": "",
            "action_items": [],
            "report": "",
        },
        config=config,
    )

    assert "Severity: high" in result["report"]
    assert "a one-paragraph summary" in result["report"]
    assert "restart the service" in result["report"]


class _FakeVoterModel:
    """Returns each of a fixed sequence of attempts in turn, one per voter call."""

    def __init__(self, attempts: list[str]) -> None:
        self._attempts = iter(attempts)

    def invoke(self, _prompt: str) -> AIMessage:
        return AIMessage(content=next(self._attempts))


def test_voting_graph_picks_the_majority_verdict(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "parallelization_agent.graph.get_chat_model",
        lambda _route: _FakeVoterModel(["yes", "no", "yes"]),
    )

    voting = build_voting_graph(checkpointer=in_memory_checkpointer, n=3)
    config = invoke_config("thread-voting")
    result = voting.invoke(
        {"prompt": "is this safe?", "attempts": [], "verdict": ""}, config=config
    )

    assert sorted(result["attempts"]) == ["no", "yes", "yes"]
    assert result["verdict"] == "yes"
