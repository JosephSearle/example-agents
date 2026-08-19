"""Unit tests for evaluator_optimizer_agent.graph — pure logic and a stubbed model, no network."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluator_optimizer_agent.graph import (
    Evaluation,
    build_evaluator_optimizer_graph,
    invoke_config,
)
from langchain_core.messages import AIMessage
import pytest

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import InMemorySaver

pytestmark = pytest.mark.unit

_STEP_PROMPTS = {
    "generate": "Write a response to the task.",
    "evaluate": "Evaluate the response against the criteria.",
}

_INITIAL_STATE = {
    "task": "some task",
    "criteria": "some criteria",
    "response": "",
    "feedback": "",
    "approved": False,
    "iteration": 0,
}


def test_invoke_config_sets_thread_id() -> None:
    config = invoke_config("some-thread-id")

    assert config["configurable"]["thread_id"] == "some-thread-id"


class _FakeEvaluator:
    """Returns a fixed sequence of `Evaluation`s, one per `evaluate` call. Stands in for
    `model.with_structured_output(Evaluation)`."""

    def __init__(self, evaluations: list[Evaluation]) -> None:
        self._evaluations = iter(evaluations)

    def invoke(self, _prompt: str) -> Evaluation:
        return next(self._evaluations)


class _FakeModel:
    """Stands in for the chat model returned by `get_chat_model`. Records every `generate`
    prompt, so tests can assert feedback actually threads into the next revision."""

    def __init__(self, evaluations: list[Evaluation]) -> None:
        self._evaluator = _FakeEvaluator(evaluations)
        self.generate_calls: list[str] = []

    def with_structured_output(self, _schema: type[Evaluation]) -> _FakeEvaluator:
        return self._evaluator

    def invoke(self, prompt: str) -> AIMessage:
        self.generate_calls.append(prompt)
        return AIMessage(content=f"draft {len(self.generate_calls)}")


def test_loop_exits_immediately_on_first_approval(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = _FakeModel([Evaluation(approved=True, feedback="")])
    monkeypatch.setattr("evaluator_optimizer_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_evaluator_optimizer_graph(
        checkpointer=in_memory_checkpointer, step_prompts=_STEP_PROMPTS
    )
    config = invoke_config("thread-approved")
    result = graph.invoke(_INITIAL_STATE, config=config)

    assert result["approved"] is True
    assert result["iteration"] == 1
    assert len(fake_model.generate_calls) == 1


def test_loop_revises_using_feedback_then_exits_on_approval(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = _FakeModel(
        [
            Evaluation(approved=False, feedback="too long"),
            Evaluation(approved=True, feedback=""),
        ]
    )
    monkeypatch.setattr("evaluator_optimizer_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_evaluator_optimizer_graph(
        checkpointer=in_memory_checkpointer, step_prompts=_STEP_PROMPTS
    )
    config = invoke_config("thread-revise")
    result = graph.invoke(_INITIAL_STATE, config=config)

    assert result["approved"] is True
    assert result["iteration"] == 2
    assert len(fake_model.generate_calls) == 2
    # The second generate call must actually thread the first evaluation's feedback in.
    assert "too long" in fake_model.generate_calls[1]
    assert "too long" not in fake_model.generate_calls[0]


def test_loop_stops_at_max_iterations_without_approval(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = _FakeModel([Evaluation(approved=False, feedback="still not right")] * 5)
    monkeypatch.setattr("evaluator_optimizer_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_evaluator_optimizer_graph(
        checkpointer=in_memory_checkpointer, step_prompts=_STEP_PROMPTS, max_iterations=2
    )
    config = invoke_config("thread-capped")
    result = graph.invoke(_INITIAL_STATE, config=config)

    assert result["approved"] is False
    assert result["iteration"] == 2
    assert len(fake_model.generate_calls) == 2
