"""Unit tests for orchestrator_workers_agent.graph — pure logic and a stubbed model, no network."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
from orchestrator_workers_agent.graph import (
    Subtask,
    TaskBreakdown,
    build_orchestrator_graph,
    invoke_config,
)
import pytest

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import InMemorySaver

pytestmark = pytest.mark.unit

_STEP_PROMPTS = {
    "orchestrate": "Break this task into subtasks.",
    "worker": "Complete this subtask.",
    "synthesize": "Combine these subtask results.",
}


def test_invoke_config_sets_thread_id() -> None:
    config = invoke_config("some-thread-id")

    assert config["configurable"]["thread_id"] == "some-thread-id"


class _FakeOrchestrator:
    """Stands in for `model.with_structured_output(TaskBreakdown)`."""

    def __init__(self, subtask_descriptions: list[str]) -> None:
        self._breakdown = TaskBreakdown(
            analysis="a fake breakdown",
            subtasks=[Subtask(description=d) for d in subtask_descriptions],
        )

    def invoke(self, _prompt: str) -> TaskBreakdown:
        return self._breakdown


class _FakeModel:
    """Stands in for the chat model returned by `get_chat_model`. Records every prompt passed
    to plain `.invoke()` (workers + synthesize), so tests can assert fan-out width and
    per-worker isolation."""

    def __init__(self, subtask_descriptions: list[str]) -> None:
        self._orchestrator = _FakeOrchestrator(subtask_descriptions)
        self.calls: list[str] = []

    def with_structured_output(self, _schema: type[TaskBreakdown]) -> _FakeOrchestrator:
        return self._orchestrator

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append(prompt)
        return AIMessage(content=f"result for: {prompt.splitlines()[-1]}")


@pytest.mark.parametrize("subtasks", [["do a"], ["do a", "do b"], ["do a", "do b", "do c", "do d"]])
def test_fan_out_spawns_one_worker_per_orchestrator_subtask(
    subtasks: list[str], in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = _FakeModel(subtasks)
    monkeypatch.setattr(
        "orchestrator_workers_agent.graph.get_chat_model", lambda _route: fake_model
    )

    graph = build_orchestrator_graph(
        checkpointer=in_memory_checkpointer, step_prompts=_STEP_PROMPTS
    )
    config = invoke_config(f"thread-{len(subtasks)}")
    result = graph.invoke(
        {
            "task": "some task",
            "analysis": "",
            "subtasks": [],
            "worker_results": [],
            "synthesis": "",
        },
        config=config,
    )

    # The graph's structure never names a subtask or a count — worker count tracks however many
    # subtasks the (fake) orchestrator decided on, and one extra call is the synthesize step.
    assert result["subtasks"] == subtasks
    assert len(result["worker_results"]) == len(subtasks)
    assert len(fake_model.calls) == len(subtasks) + 1


def test_synthesize_runs_once_after_all_workers_complete(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = _FakeModel(["do a", "do b"])
    monkeypatch.setattr(
        "orchestrator_workers_agent.graph.get_chat_model", lambda _route: fake_model
    )

    graph = build_orchestrator_graph(
        checkpointer=in_memory_checkpointer, step_prompts=_STEP_PROMPTS
    )
    config = invoke_config("thread-synthesis")
    result = graph.invoke(
        {
            "task": "some task",
            "analysis": "",
            "subtasks": [],
            "worker_results": [],
            "synthesis": "",
        },
        config=config,
    )

    # Two worker calls + one synthesize call; synthesize's own prompt (not either worker's) is
    # the last one made, and it references both prior worker results.
    synthesize_calls = [
        call for call in fake_model.calls if call.startswith(_STEP_PROMPTS["synthesize"])
    ]
    assert len(synthesize_calls) == 1
    assert "Subtask result 1" in synthesize_calls[0]
    assert "Subtask result 2" in synthesize_calls[0]
    assert result["synthesis"] != ""


def test_each_worker_only_sees_its_own_subtask(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = _FakeModel(["do a", "do b"])
    monkeypatch.setattr(
        "orchestrator_workers_agent.graph.get_chat_model", lambda _route: fake_model
    )

    graph = build_orchestrator_graph(
        checkpointer=in_memory_checkpointer, step_prompts=_STEP_PROMPTS
    )
    config = invoke_config("thread-isolation")
    graph.invoke(
        {
            "task": "some task",
            "analysis": "",
            "subtasks": [],
            "worker_results": [],
            "synthesis": "",
        },
        config=config,
    )

    worker_calls = [call for call in fake_model.calls if call.startswith(_STEP_PROMPTS["worker"])]
    called_subtasks = {call.splitlines()[-1] for call in worker_calls}
    assert called_subtasks == {"do a", "do b"}
