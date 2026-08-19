"""Integration-test fixtures — require `docker compose up -d postgres` (see repo README)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents_common import get_settings
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


class _FakeChatModel:
    """Always returns a canned joke. Stands in for the real gateway model so checkpointing tests
    exercise real Postgres without depending on network access to a live model."""

    def invoke(self, prompt: str) -> AIMessage:
        topic = prompt.splitlines()[-1]
        return AIMessage(content=f"a joke about {topic}")


@pytest.fixture
def fake_chat_model(monkeypatch: pytest.MonkeyPatch) -> _FakeChatModel:
    """Patches `map_reduce_agent.graph.get_chat_model` so `build_map_reduce_graph()` never
    reaches the network."""
    model = _FakeChatModel()
    monkeypatch.setattr("map_reduce_agent.graph.get_chat_model", lambda _route: model)
    return model


@pytest.fixture
def joke_prompt() -> str:
    """A hermetic worker prompt, so tests don't depend on a live MLflow prompt registry."""
    return "Write a short joke about the given topic."
