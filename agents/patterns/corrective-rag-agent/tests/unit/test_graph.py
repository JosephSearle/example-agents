"""Unit tests for corrective_rag_agent.graph — pure logic, a stubbed model (grading, rewriting,
and generation all faked), and a fake retriever, no network, no real Milvus."""

from __future__ import annotations

from typing import TYPE_CHECKING

from corrective_rag_agent.graph import (
    MAX_RETRIES,
    NO_CONTEXT_ANSWER,
    build_rag_graph,
    invoke_config,
)
from langchain_core.messages import AIMessage
import pytest

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import InMemorySaver

pytestmark = pytest.mark.unit

_PROMPTS = {
    "grade_documents": "Grade this document.",
    "transform_query": "Rewrite the question.",
    "generate": "Answer using only the context.",
}


class _FakeDocument:
    def __init__(self, page_content: str) -> None:
        self.page_content = page_content


class _FakeRetriever:
    """Returns a fixed set of documents regardless of the query, and records every query."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[_FakeDocument]:
        self.queries.append(query)
        return [_FakeDocument(chunk) for chunk in self._chunks]


class _FakeGrade:
    def __init__(self, binary_score: str) -> None:
        self.binary_score = binary_score


class _FakeGrader:
    """Stands in for `model.with_structured_output(DocumentGrade)`. Grades every document "yes"
    if `all_relevant`, else "no" — realistic enough to drive the routing logic under test."""

    def __init__(self, all_relevant: bool) -> None:
        self._all_relevant = all_relevant
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> _FakeGrade:
        self.calls.append(prompt)
        return _FakeGrade("yes" if self._all_relevant else "no")


class _FakeModel:
    """Stands in for the chat model returned by `get_chat_model` — serves grading (via
    `with_structured_output`), query rewriting, and generation, distinguishing the latter two by
    a marker each node's prompt is known to include."""

    def __init__(
        self, *, all_relevant: bool, rewritten_question: str, generated_answer: str
    ) -> None:
        self._grader = _FakeGrader(all_relevant)
        self._rewritten_question = rewritten_question
        self._generated_answer = generated_answer
        self.invoke_calls: list[str] = []

    def with_structured_output(self, _schema: object) -> _FakeGrader:
        return self._grader

    def invoke(self, prompt: str) -> AIMessage:
        self.invoke_calls.append(prompt)
        if "Original question:" in prompt:
            return AIMessage(content=self._rewritten_question)
        return AIMessage(content=self._generated_answer)


def _initial_state(question: str) -> dict[str, object]:
    return {
        "question": question,
        "original_question": question,
        "documents": [],
        "documents_sufficient": False,
        "retry_count": 0,
        "answer": "",
    }


def test_invoke_config_sets_thread_id() -> None:
    config = invoke_config("some-thread-id")

    assert config["configurable"]["thread_id"] == "some-thread-id"


def test_all_relevant_documents_generates_without_retrying(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retriever = _FakeRetriever(["Milvus stores vectors."])
    fake_model = _FakeModel(
        all_relevant=True, rewritten_question="unused", generated_answer="a grounded answer"
    )
    monkeypatch.setattr("corrective_rag_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer, prompts=_PROMPTS, retriever=fake_retriever
    )
    result = graph.invoke(_initial_state("What is Milvus?"), config=invoke_config("thread-1"))

    assert result["answer"] == "a grounded answer"
    assert result["retry_count"] == 0
    assert fake_retriever.queries == ["What is Milvus?"]


def test_irrelevant_documents_trigger_a_retry_that_then_succeeds(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retriever = _FakeRetriever(["some chunk"])
    fake_model = _FakeModel(
        all_relevant=False, rewritten_question="a better question", generated_answer="never reached"
    )
    monkeypatch.setattr("corrective_rag_agent.graph.get_chat_model", lambda _route: fake_model)
    # Force the second grading pass to succeed so the loop terminates after exactly one retry.
    grade_calls = {"count": 0}
    original_invoke = fake_model._grader.invoke

    def flaky_invoke(prompt: str) -> _FakeGrade:
        grade_calls["count"] += 1
        if grade_calls["count"] > 1:
            return _FakeGrade("yes")
        return original_invoke(prompt)

    fake_model._grader.invoke = flaky_invoke  # type: ignore[method-assign]

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer, prompts=_PROMPTS, retriever=fake_retriever
    )
    result = graph.invoke(_initial_state("vague question"), config=invoke_config("thread-2"))

    assert result["retry_count"] == 1
    assert fake_retriever.queries == ["vague question", "a better question"]
    assert result["answer"] == "never reached"  # the (only) generate call used the canned answer


def test_retry_cap_reached_generates_anyway_instead_of_looping_forever(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retriever = _FakeRetriever(["always irrelevant chunk"])
    fake_model = _FakeModel(
        all_relevant=False, rewritten_question="still vague", generated_answer="best effort answer"
    )
    monkeypatch.setattr("corrective_rag_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer,
        prompts=_PROMPTS,
        retriever=fake_retriever,
        max_retries=MAX_RETRIES,
    )
    result = graph.invoke(
        _initial_state("persistently bad question"), config=invoke_config("thread-3")
    )

    # Bounded: retrieve runs once up-front, then once per retry — never loops forever.
    assert len(fake_retriever.queries) == MAX_RETRIES + 1
    assert result["retry_count"] == MAX_RETRIES
    # Every retrieval was graded fully irrelevant, so `documents` is empty even at the cap —
    # `generate` correctly takes the honest "no context" branch rather than fabricating from
    # documents it already graded irrelevant. The `decide_to_generate` cap guard is still what
    # got us to `generate` at all instead of looping a third time (see the retry_count/queries
    # assertions above) — this assertion just confirms `generate` itself behaves honestly once
    # reached with nothing relevant in hand.
    assert result["answer"] == NO_CONTEXT_ANSWER
