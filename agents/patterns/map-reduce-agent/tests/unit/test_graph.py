"""Unit tests for map_reduce_agent.graph — pure logic and a stubbed model, no network."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
from map_reduce_agent.graph import build_map_reduce_graph, invoke_config
import pytest

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import InMemorySaver

pytestmark = pytest.mark.unit

_JOKE_PROMPT = "Write a short joke about the given topic."


def test_invoke_config_sets_thread_id() -> None:
    config = invoke_config("some-thread-id")

    assert config["configurable"]["thread_id"] == "some-thread-id"


class _FakeModel:
    """Stands in for the chat model returned by `get_chat_model`. Records every prompt it's
    called with, so tests can assert one worker call happened per topic."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append(prompt)
        return AIMessage(content=f"a joke about {prompt.splitlines()[-1]}")


@pytest.mark.parametrize("topics", [[], ["cats"], ["cats", "airports"], ["a", "b", "c", "d"]])
def test_fan_out_spawns_one_worker_per_topic(
    topics: list[str], in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = _FakeModel()
    monkeypatch.setattr("map_reduce_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_map_reduce_graph(checkpointer=in_memory_checkpointer, joke_prompt=_JOKE_PROMPT)
    config = invoke_config(f"thread-{len(topics)}")
    result = graph.invoke({"topics": topics, "jokes": [], "summary": ""}, config=config)

    # The graph's structure never names a topic or a count — the number of generate_joke workers
    # (and therefore model calls) tracks len(topics) at invoke time, per topics.
    assert len(fake_model.calls) == len(topics)
    assert len(result["jokes"]) == len(topics)


def test_combine_jokes_reports_the_correct_count(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("map_reduce_agent.graph.get_chat_model", lambda _route: _FakeModel())

    graph = build_map_reduce_graph(checkpointer=in_memory_checkpointer, joke_prompt=_JOKE_PROMPT)
    config = invoke_config("thread-combine")
    result = graph.invoke(
        {"topics": ["cats", "airports", "Mondays"], "jokes": [], "summary": ""}, config=config
    )

    assert "Generated 3 joke(s)" in result["summary"]
    assert len(result["jokes"]) == 3


def test_each_worker_only_sees_its_own_topic(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = _FakeModel()
    monkeypatch.setattr("map_reduce_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_map_reduce_graph(checkpointer=in_memory_checkpointer, joke_prompt=_JOKE_PROMPT)
    config = invoke_config("thread-isolation")
    graph.invoke({"topics": ["cats", "airports"], "jokes": [], "summary": ""}, config=config)

    called_topics = {call.splitlines()[-1].removeprefix("Topic: ") for call in fake_model.calls}
    assert called_topics == {"cats", "airports"}
