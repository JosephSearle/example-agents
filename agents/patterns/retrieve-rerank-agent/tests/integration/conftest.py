"""Integration-test fixtures — require `docker compose up -d postgres milvus-standalone` (see
repo README) and the seeded `basic_rag_agent` Milvus collection (`make provision-milvus-collections`).
No real reranker HTTP call — the reranker endpoint isn't dockerized/local, so a fake reranker
stands in here (see this package's README); a real reranker call is only exercised in
`tests/evals`."""

from __future__ import annotations

from typing import TYPE_CHECKING

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


class _FakeChatModel:
    """Always returns a canned response. Stands in for the real gateway chat model so
    checkpointing tests exercise real Postgres + real Milvus retrieval without depending on
    network access to a live chat model."""

    def invoke(self, _prompt: str) -> AIMessage:
        return AIMessage(content="a canned grounded answer")


@pytest.fixture
def fake_chat_model(monkeypatch: pytest.MonkeyPatch) -> _FakeChatModel:
    """Patches `retrieve_rerank_agent.graph.get_chat_model` so `build_rag_graph()` never reaches
    the network for generation — retrieval still hits the real, seeded Milvus collection."""
    model = _FakeChatModel()
    monkeypatch.setattr("retrieve_rerank_agent.graph.get_chat_model", lambda _route: model)
    return model


class _FakeReranker:
    """Passes the candidate set through unchanged (up to `top_n`) — the point of this fixture is
    exercising real Milvus retrieval end-to-end, not reranking quality, which is an evals concern."""

    def rerank(self, _query: str, documents: list[str], *, top_n: int) -> list[dict[str, object]]:
        return [{"index": i, "score": 1.0} for i in range(len(documents))][:top_n]


@pytest.fixture
def fake_reranker() -> _FakeReranker:
    """A reranker that doesn't make a real HTTP call — see this module's docstring."""
    return _FakeReranker()


@pytest.fixture
def rag_prompt() -> str:
    """Hermetic generation prompt, so tests don't depend on a live MLflow prompt registry."""
    return "Answer using only the context below."
