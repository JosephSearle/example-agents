"""Unit tests for network_mesh_agent.graph — pure logic and a stubbed model, no network."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
from network_mesh_agent.graph import (
    Critique,
    ResearchFinding,
    build_mesh_graph,
    invoke_config,
)
import pytest

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import InMemorySaver

pytestmark = pytest.mark.unit

_AGENT_PROMPTS = {
    "researcher": "Research the task.",
    "critic": "Critique the finding.",
    "writer": "Write the final answer.",
}


def _initial_state(task: str) -> dict[str, object]:
    return {
        "task": task,
        "messages": [],
        "needs_critique": False,
        "needs_more_research": False,
        "research_rounds": 0,
        "final_answer": "",
    }


def test_invoke_config_sets_thread_id() -> None:
    config = invoke_config("some-thread-id")

    assert config["configurable"]["thread_id"] == "some-thread-id"


class _FakeStructuredModel:
    """Stands in for `model.with_structured_output(...)`, playing back one canned result."""

    def __init__(self, result: ResearchFinding | Critique) -> None:
        self._result = result

    def invoke(self, _prompt: str) -> ResearchFinding | Critique:
        return self._result


class _FakeModel:
    """Stands in for the chat model `get_chat_model` returns.

    `_researcher_results`/`_critic_results` are consumed one per call, in order — lets a test
    script a researcher<->critic loop before the mesh converges on the writer.
    """

    def __init__(
        self,
        researcher_results: list[ResearchFinding],
        critic_results: list[Critique],
        writer_answer: str = "the final answer",
    ) -> None:
        self._researcher_results = list(researcher_results)
        self._critic_results = list(critic_results)
        self._writer_answer = writer_answer
        self.writer_calls: list[str] = []

    def with_structured_output(self, schema: type) -> _FakeStructuredModel:
        if schema is ResearchFinding:
            return _FakeStructuredModel(self._researcher_results.pop(0))
        return _FakeStructuredModel(self._critic_results.pop(0))

    def invoke(self, prompt: str) -> AIMessage:
        self.writer_calls.append(prompt)
        return AIMessage(content=self._writer_answer)


def test_researcher_goes_straight_to_writer_when_no_critique_needed(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = _FakeModel(
        researcher_results=[ResearchFinding(finding="solid finding", needs_critique=False)],
        critic_results=[],
    )
    monkeypatch.setattr("network_mesh_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_mesh_graph(checkpointer=in_memory_checkpointer, agent_prompts=_AGENT_PROMPTS)
    result = graph.invoke(_initial_state("some task"), config=invoke_config("thread-direct"))

    assert result["final_answer"] == "the final answer"
    assert len(fake_model.writer_calls) == 1
    assert not any(m["role"] == "critic" for m in result["messages"])


def test_researcher_routes_to_critic_when_critique_needed(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = _FakeModel(
        researcher_results=[ResearchFinding(finding="thin finding", needs_critique=True)],
        critic_results=[Critique(critique="looks fine", needs_more_research=False)],
    )
    monkeypatch.setattr("network_mesh_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_mesh_graph(checkpointer=in_memory_checkpointer, agent_prompts=_AGENT_PROMPTS)
    result = graph.invoke(_initial_state("some task"), config=invoke_config("thread-critique"))

    assert any(m == {"role": "critic", "content": "looks fine"} for m in result["messages"])
    assert result["final_answer"] == "the final answer"


def test_critic_routes_back_to_researcher_when_more_research_needed(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = _FakeModel(
        researcher_results=[
            ResearchFinding(finding="round one", needs_critique=True),
            ResearchFinding(finding="round two", needs_critique=False),
        ],
        critic_results=[Critique(critique="needs more depth", needs_more_research=True)],
    )
    monkeypatch.setattr("network_mesh_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_mesh_graph(
        checkpointer=in_memory_checkpointer, agent_prompts=_AGENT_PROMPTS, max_research_rounds=5
    )
    result = graph.invoke(_initial_state("some task"), config=invoke_config("thread-loop"))

    # Two researcher rounds, one critic round, then straight to the writer — the route-back edge
    # actually fired, and the mesh still converged.
    assert result["research_rounds"] == 2
    assert [m["role"] for m in result["messages"]] == [
        "researcher",
        "critic",
        "researcher",
        "writer",
    ]


def test_max_research_rounds_forces_convergence_to_writer(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The critic always asks for more research, but the round cap must win regardless.
    fake_model = _FakeModel(
        researcher_results=[
            ResearchFinding(finding="round one", needs_critique=True),
            ResearchFinding(finding="round two", needs_critique=True),
        ],
        critic_results=[
            Critique(critique="still not enough", needs_more_research=True),
        ],
    )
    monkeypatch.setattr("network_mesh_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_mesh_graph(
        checkpointer=in_memory_checkpointer, agent_prompts=_AGENT_PROMPTS, max_research_rounds=2
    )
    result = graph.invoke(_initial_state("some task"), config=invoke_config("thread-capped"))

    assert result["research_rounds"] == 2
    assert result["final_answer"] == "the final answer"
