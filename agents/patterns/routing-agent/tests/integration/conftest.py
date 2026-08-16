"""Integration-test fixtures — require `docker compose up -d postgres` (see repo README)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents_common import get_settings
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
import pytest
from routing_agent.graph import TicketCategory

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


class _FakeClassifier:
    """Always classifies a ticket as "technical". Stands in for
    `model.with_structured_output(TicketCategory)`."""

    def invoke(self, _message: str) -> TicketCategory:
        return TicketCategory(category="technical")


class _FakeChatModel:
    """Always classifies as "technical" and returns a canned handler response. Stands in for
    the real gateway model so checkpointing tests exercise real Postgres without depending on
    network access to a live model."""

    def with_structured_output(self, _schema: type[TicketCategory]) -> _FakeClassifier:
        return _FakeClassifier()

    def invoke(self, _prompt: str) -> AIMessage:
        return AIMessage(content="diagnostic steps here")


@pytest.fixture
def fake_chat_model(monkeypatch: pytest.MonkeyPatch) -> _FakeChatModel:
    """Patches `routing_agent.graph.get_chat_model` so `build_router()` never reaches the network."""
    model = _FakeChatModel()
    monkeypatch.setattr("routing_agent.graph.get_chat_model", lambda _route: model)
    return model


@pytest.fixture
def route_prompts() -> dict[str, str]:
    """Hermetic handler prompts, so tests don't depend on a live MLflow prompt registry."""
    return {
        "general": "Answer this general support question.",
        "refund": "Follow the refund policy strictly.",
        "technical": "Provide a technical troubleshooting response.",
    }
