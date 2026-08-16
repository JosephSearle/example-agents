"""Unit tests for routing_agent.graph — pure logic and a stubbed model, no network."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
import pytest
from routing_agent.graph import TicketCategory, build_router, invoke_config

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import InMemorySaver

pytestmark = pytest.mark.unit

_ROUTE_PROMPTS = {
    "general": "Answer this general support question.",
    "refund": "Follow the refund policy strictly.",
    "technical": "Provide a technical troubleshooting response.",
}


def test_invoke_config_sets_thread_id() -> None:
    config = invoke_config("some-thread-id")

    assert config["configurable"]["thread_id"] == "some-thread-id"


class _FakeClassifier:
    """Stands in for `model.with_structured_output(TicketCategory)`."""

    def __init__(self, category: str) -> None:
        self._category = category

    def invoke(self, _message: str) -> TicketCategory:
        return TicketCategory(category=self._category)  # type: ignore[arg-type]


class _FakeModel:
    """Stands in for the chat model returned by `get_chat_model`."""

    def __init__(self, category: str, response_text: str) -> None:
        self._classifier = _FakeClassifier(category)
        self._response_text = response_text

    def with_structured_output(self, _schema: type[TicketCategory]) -> _FakeClassifier:
        return self._classifier

    def invoke(self, _prompt: str) -> AIMessage:
        return AIMessage(content=self._response_text)


@pytest.mark.parametrize("category", ["general", "refund", "technical"])
def test_router_dispatches_to_the_classified_category(
    category: str, in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "routing_agent.graph.get_chat_model",
        lambda _route: _FakeModel(category, f"handled as {category}"),
    )

    router = build_router(checkpointer=in_memory_checkpointer, route_prompts=_ROUTE_PROMPTS)
    config = invoke_config(f"thread-{category}")
    result = router.invoke(
        {"message": "some ticket", "category": "", "response": ""}, config=config
    )

    assert result["category"] == category
    assert result["response"] == f"handled as {category}"


def test_router_only_runs_the_matched_handler(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class _TrackingModel(_FakeModel):
        def invoke(self, prompt: str) -> AIMessage:
            calls.append(prompt)
            return super().invoke(prompt)

    monkeypatch.setattr(
        "routing_agent.graph.get_chat_model",
        lambda _route: _TrackingModel("technical", "diagnostic steps here"),
    )

    router = build_router(checkpointer=in_memory_checkpointer, route_prompts=_ROUTE_PROMPTS)
    config = invoke_config("thread-single-handler")
    router.invoke({"message": "it crashes", "category": "", "response": ""}, config=config)

    assert len(calls) == 1
    assert _ROUTE_PROMPTS["technical"] in calls[0]
