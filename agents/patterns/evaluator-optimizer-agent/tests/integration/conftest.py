"""Integration-test fixtures — require `docker compose up -d postgres` (see repo README)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents_common import get_settings
from evaluator_optimizer_agent.graph import Evaluation
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def postgres_checkpointer() -> Iterator[PostgresSaver]:
    """A real PostgresSaver against the docker-compose Postgres instance.

    Skips (rather than fails) if `Settings.postgres_uri` isn't reachable, so `pytest -m
    integration` fails loudly in CI (where the service container is always up) but doesn't
    block a laptop run that hasn't started docker compose.
    """
    uri = get_settings().postgres_uri
    try:
        with PostgresSaver.from_conn_string(uri) as checkpointer:
            checkpointer.setup()
            yield checkpointer
    except Exception as exc:
        pytest.skip(f"Postgres not reachable at {uri}: {exc}")


class _FakeEvaluator:
    """Approves on the second call, never on the first. Stands in for
    `model.with_structured_output(Evaluation)`."""

    def __init__(self) -> None:
        self._calls = 0

    def invoke(self, _prompt: str) -> Evaluation:
        self._calls += 1
        if self._calls == 1:
            return Evaluation(approved=False, feedback="needs work")
        return Evaluation(approved=True, feedback="")


class _FakeChatModel:
    """Always produces a canned draft and approves on the second evaluation. Stands in for the
    real gateway model so checkpointing tests exercise real Postgres without depending on
    network access to a live model."""

    def __init__(self) -> None:
        self._evaluator = _FakeEvaluator()

    def with_structured_output(self, _schema: type[Evaluation]) -> _FakeEvaluator:
        return self._evaluator

    def invoke(self, _prompt: str) -> AIMessage:
        return AIMessage(content="a draft response")


@pytest.fixture
def fake_chat_model(monkeypatch: pytest.MonkeyPatch) -> _FakeChatModel:
    """Patches `evaluator_optimizer_agent.graph.get_chat_model` so
    `build_evaluator_optimizer_graph()` never reaches the network."""
    model = _FakeChatModel()
    monkeypatch.setattr("evaluator_optimizer_agent.graph.get_chat_model", lambda _route: model)
    return model


@pytest.fixture
def step_prompts() -> dict[str, str]:
    """Hermetic step prompts, so tests don't depend on a live MLflow prompt registry."""
    return {
        "generate": "Write a response to the task.",
        "evaluate": "Evaluate the response against the criteria.",
    }
