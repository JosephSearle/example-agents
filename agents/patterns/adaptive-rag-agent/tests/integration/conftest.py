"""Integration-test fixtures — require `docker compose up -d postgres milvus-standalone` (see
repo README) and the seeded `basic_rag_agent` Milvus collection (`make provision-milvus-collections`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from adaptive_rag_agent.graph import ComplexityRoute, DocumentGrade, SubQuestions
from agents_common import get_settings
from basic_rag_agent.graph import COLLECTION_NAME
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
from pymilvus import MilvusClient
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


@pytest.fixture
def real_milvus_collection() -> None:
    """Skips (rather than fails) if the seeded `basic_rag_agent` Milvus collection isn't
    reachable — same "skip on a laptop that hasn't started the service, fail loudly in CI"
    convention as `postgres_checkpointer`."""
    settings = get_settings()
    try:
        client = MilvusClient(uri=settings.milvus_uri)
        has_collection = client.has_collection(COLLECTION_NAME)
        client.close()
    except Exception as exc:
        pytest.skip(f"Milvus not reachable at {settings.milvus_uri}: {exc}")
    if not has_collection:
        pytest.skip(
            f"Collection '{COLLECTION_NAME}' not seeded — run `make provision-milvus-collections`"
        )


class _FakeRoute:
    complexity = "single_step"


class _FakeGrade:
    binary_score = "yes"


class _FakeStructuredCaller:
    def __init__(self, result: object) -> None:
        self._result = result

    def invoke(self, _prompt: str) -> object:
        return self._result


class _FakeChatModel:
    """Always routes to single_step and returns a canned response. Stands in for the real
    gateway chat model so checkpointing tests exercise real Postgres + real Milvus retrieval
    without depending on network access to a live chat model."""

    def with_structured_output(self, schema: object) -> _FakeStructuredCaller:
        if schema is ComplexityRoute:
            return _FakeStructuredCaller(_FakeRoute())
        if schema is DocumentGrade:
            return _FakeStructuredCaller(_FakeGrade())
        if schema is SubQuestions:
            # Constructed unconditionally by build_rag_graph even though this fixture always
            # routes to single_step — never actually invoked in that branch.
            return _FakeStructuredCaller(None)
        raise AssertionError(f"unexpected schema: {schema}")

    def invoke(self, _prompt: str) -> AIMessage:
        return AIMessage(content="a canned grounded answer")


@pytest.fixture
def fake_chat_model(monkeypatch: pytest.MonkeyPatch) -> _FakeChatModel:
    """Patches `adaptive_rag_agent.graph.get_chat_model` so `build_rag_graph()` never reaches the
    network — retrieval still hits the real, seeded Milvus collection (single_step branch)."""
    model = _FakeChatModel()
    monkeypatch.setattr("adaptive_rag_agent.graph.get_chat_model", lambda _route: model)
    return model


@pytest.fixture
def prompts() -> dict[str, str]:
    """Hermetic prompts, so tests don't depend on a live MLflow prompt registry."""
    return {
        "route": "Classify the question.",
        "generate_direct": "Answer directly.",
        "grade_documents": "Grade this document.",
        "transform_query": "Rewrite the question.",
        "generate": "Answer using only the context.",
        "decompose": "Break the question into sub-questions.",
        "synthesize": "Combine the sub-answers.",
    }
