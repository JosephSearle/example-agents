"""Integration-test fixtures — require `docker compose up -d postgres` (see repo README)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents_common import get_settings
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
from parallelization_agent.graph import ActionItems, SeverityAssessment
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


class _FakeStructuredModel:
    def __init__(self, result: SeverityAssessment | ActionItems) -> None:
        self._result = result

    def invoke(self, _prompt: str) -> SeverityAssessment | ActionItems:
        return self._result


class _FakeChatModel:
    """Always returns canned section outputs. Stands in for the real gateway model so
    checkpointing tests exercise real Postgres without depending on network access to a live
    model."""

    def with_structured_output(
        self, schema: type[SeverityAssessment] | type[ActionItems]
    ) -> _FakeStructuredModel:
        if schema is SeverityAssessment:
            return _FakeStructuredModel(SeverityAssessment(severity="critical"))
        return _FakeStructuredModel(ActionItems(items=["page the on-call engineer"]))

    def invoke(self, _prompt: str) -> AIMessage:
        return AIMessage(content="the database ran out of disk space")


@pytest.fixture
def fake_chat_model(monkeypatch: pytest.MonkeyPatch) -> _FakeChatModel:
    """Patches `parallelization_agent.graph.get_chat_model` so `build_sectioning_graph()` never
    reaches the network."""
    model = _FakeChatModel()
    monkeypatch.setattr("parallelization_agent.graph.get_chat_model", lambda _route: model)
    return model


@pytest.fixture
def section_prompts() -> dict[str, str]:
    """Hermetic section prompts, so tests don't depend on a live MLflow prompt registry."""
    return {
        "summarize": "Summarize this incident in one paragraph.",
        "assess_severity": "Assess the severity of this incident.",
        "extract_action_items": "Extract follow-up action items from this incident.",
    }
