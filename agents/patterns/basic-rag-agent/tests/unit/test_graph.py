"""Unit tests for basic_rag_agent.graph — pure logic, a stubbed model, and a fake retriever, no
network, no real Milvus."""

from __future__ import annotations

from typing import TYPE_CHECKING

from basic_rag_agent.graph import NO_CONTEXT_ANSWER, build_rag_graph, invoke_config
from langchain_core.messages import AIMessage
import pytest

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import InMemorySaver

pytestmark = pytest.mark.unit

_RAG_PROMPT = "Answer using only the context below."


class _FakeDocument:
    def __init__(self, page_content: str) -> None:
        self.page_content = page_content


class _FakeRetriever:
    """Stands in for `langchain_milvus.Milvus(...).as_retriever()`. Returns a fixed set of
    documents (or none) regardless of the query, and records every query it was called with."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[_FakeDocument]:
        self.queries.append(query)
        return [_FakeDocument(chunk) for chunk in self._chunks]


class _FakeModel:
    """Stands in for the chat model returned by `get_chat_model`."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append(prompt)
        return AIMessage(content=self._response_text)


def _initial_state(question: str) -> dict[str, object]:
    return {"question": question, "retrieved_chunks": [], "answer": ""}


def test_invoke_config_sets_thread_id() -> None:
    config = invoke_config("some-thread-id")

    assert config["configurable"]["thread_id"] == "some-thread-id"


def test_generate_stuffs_retrieved_chunks_into_the_prompt(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retriever = _FakeRetriever(["Milvus stores vectors.", "Attu is Milvus's web UI."])
    fake_model = _FakeModel("Milvus stores vectors and Attu is its web UI.")
    monkeypatch.setattr("basic_rag_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer, rag_prompt=_RAG_PROMPT, retriever=fake_retriever
    )
    result = graph.invoke(
        _initial_state("What is Milvus?"), config=invoke_config("thread-grounded")
    )

    assert result["retrieved_chunks"] == [
        "Milvus stores vectors.",
        "Attu is Milvus's web UI.",
    ]
    assert result["answer"] == "Milvus stores vectors and Attu is its web UI."
    assert fake_retriever.queries == ["What is Milvus?"]
    assert "Milvus stores vectors." in fake_model.calls[0]
    assert "Attu is Milvus's web UI." in fake_model.calls[0]
    assert "What is Milvus?" in fake_model.calls[0]


def test_generate_returns_no_context_answer_without_calling_the_model_on_empty_retrieval(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retriever = _FakeRetriever([])
    fake_model = _FakeModel("this should never be returned")
    monkeypatch.setattr("basic_rag_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer, rag_prompt=_RAG_PROMPT, retriever=fake_retriever
    )
    result = graph.invoke(
        _initial_state("What's the airspeed velocity of an unladen swallow?"),
        config=invoke_config("thread-empty"),
    )

    assert result["answer"] == NO_CONTEXT_ANSWER
    assert fake_model.calls == []
