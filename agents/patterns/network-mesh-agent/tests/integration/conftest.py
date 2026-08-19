"""Integration-test fixtures — require `docker compose up -d postgres` (see repo README)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents_common import get_settings
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
from network_mesh_agent.graph import Critique, ResearchFinding
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def postgres_checkpointer() -> Iterator[PostgresSaver]:
    """A real PostgresSaver against the docker-compose Postgres instance.

    Skips (rather than fails) if `Settings.postgres_uri` isn't reachable, so `pytest -m
    integration` fails loudly in CI (where the service container is always up) but doesn't block
    a laptop run that hasn't started docker compose.
    """
    uri = get_settings().postgres_uri
    try:
        with PostgresSaver.from_conn_string(uri) as checkpointer:
            checkpointer.setup()
            yield checkpointer
    except Exception as exc:
        pytest.skip(f"Postgres not reachable at {uri}: {exc}")


class _FakeStructuredModel:
    def __init__(self, result: ResearchFinding | Critique) -> None:
        self._result = result

    def invoke(self, _prompt: str) -> ResearchFinding | Critique:
        return self._result


class _FakeChatModel:
    """Always decides the finding is fine as-is (no critique round). Stands in for the real
    gateway model so checkpointing tests exercise real Postgres without depending on network
    access to a live model."""

    def with_structured_output(self, schema: type) -> _FakeStructuredModel:
        if schema is ResearchFinding:
            return _FakeStructuredModel(
                ResearchFinding(finding="a fake finding", needs_critique=False)
            )
        return _FakeStructuredModel(Critique(critique="a fake critique", needs_more_research=False))

    def invoke(self, _prompt: str) -> AIMessage:
        return AIMessage(content="a canned final answer")


@pytest.fixture
def fake_chat_model(monkeypatch: pytest.MonkeyPatch) -> _FakeChatModel:
    """Patches `network_mesh_agent.graph.get_chat_model` so `build_mesh_graph()` never reaches the
    network."""
    model = _FakeChatModel()
    monkeypatch.setattr("network_mesh_agent.graph.get_chat_model", lambda _route: model)
    return model


@pytest.fixture
def agent_prompts() -> dict[str, str]:
    """Hermetic peer prompts, so tests don't depend on a live MLflow prompt registry."""
    return {
        "researcher": "Research the task.",
        "critic": "Critique the finding.",
        "writer": "Write the final answer.",
    }
