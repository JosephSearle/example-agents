"""Unit tests for retrieve_rerank_agent.graph — pure logic, a stubbed model, a fake retriever,
and a fake reranker, no network, no real Milvus, no real reranker HTTP call."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
import pytest
from retrieve_rerank_agent.graph import NO_CONTEXT_ANSWER, build_rag_graph, invoke_config

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import InMemorySaver

pytestmark = pytest.mark.unit

_RAG_PROMPT = "Answer using only the context below."


class _FakeDocument:
    def __init__(self, page_content: str) -> None:
        self.page_content = page_content


class _FakeRetriever:
    """Returns a fixed set of candidate documents regardless of the query, and records every
    query it was called with."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[_FakeDocument]:
        self.queries.append(query)
        return [_FakeDocument(chunk) for chunk in self._chunks]


class _FakeReranker:
    """Returns a fixed ranking (by index into whatever `documents` it was called with),
    regardless of `query`, and records every call it received."""

    def __init__(self, order: list[int]) -> None:
        self._order = order
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[dict[str, object]]:
        self.calls.append((query, documents, top_n))
        return [{"index": i, "score": 1.0 - 0.1 * rank} for rank, i in enumerate(self._order)][
            :top_n
        ]


class _FakeModel:
    """Stands in for the chat model returned by `get_chat_model`."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append(prompt)
        return AIMessage(content=self._response_text)


def _initial_state(question: str) -> dict[str, object]:
    return {"question": question, "candidate_chunks": [], "reranked_chunks": [], "answer": ""}


def test_invoke_config_sets_thread_id() -> None:
    config = invoke_config("some-thread-id")

    assert config["configurable"]["thread_id"] == "some-thread-id"


def test_rerank_narrows_and_reorders_candidates_before_generation(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = ["least relevant chunk", "most relevant chunk", "middling chunk"]
    fake_retriever = _FakeRetriever(candidates)
    # Reranker picks index 1 ("most relevant") first, then index 2 — deliberately not retrieval
    # order, to prove `reranked_chunks` reflects the reranker's order, not the retriever's.
    fake_reranker = _FakeReranker(order=[1, 2, 0])
    fake_model = _FakeModel("a grounded answer")
    monkeypatch.setattr("retrieve_rerank_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer,
        rag_prompt=_RAG_PROMPT,
        retriever=fake_retriever,
        reranker=fake_reranker,
        top_n=2,
    )
    result = graph.invoke(_initial_state("What is Milvus?"), config=invoke_config("thread-rerank"))

    assert result["candidate_chunks"] == candidates
    assert result["reranked_chunks"] == ["most relevant chunk", "middling chunk"]
    assert result["answer"] == "a grounded answer"
    assert fake_reranker.calls == [("What is Milvus?", candidates, 2)]
    assert "most relevant chunk" in fake_model.calls[0]
    assert "least relevant chunk" not in fake_model.calls[0]


def test_generate_returns_no_context_answer_without_calling_reranker_or_model_on_empty_retrieval(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retriever = _FakeRetriever([])
    fake_reranker = _FakeReranker(order=[])
    fake_model = _FakeModel("this should never be returned")
    monkeypatch.setattr("retrieve_rerank_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer,
        rag_prompt=_RAG_PROMPT,
        retriever=fake_retriever,
        reranker=fake_reranker,
    )
    result = graph.invoke(
        _initial_state("What's the airspeed velocity of an unladen swallow?"),
        config=invoke_config("thread-empty"),
    )

    assert result["answer"] == NO_CONTEXT_ANSWER
    assert fake_reranker.calls == []
    assert fake_model.calls == []
