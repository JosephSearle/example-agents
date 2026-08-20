"""Unit tests for self_rag_agent.graph — pure logic, a fully stubbed model (document grading,
query rewriting, generation, hallucination grading, and usefulness grading all faked), and a fake
retriever, no network, no real Milvus.

Covers the four cases this pattern's own "infinite-loop risk" caveat calls for: first-pass
success, one regenerate loop, one re-retrieve loop, and both caps reached (bounded, doesn't loop
forever)."""

from __future__ import annotations

from itertools import cycle, repeat
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
import pytest
from self_rag_agent.graph import (
    MAX_REGENERATE,
    MAX_RETRIES,
    AnswerUsefulnessGrade,
    DocumentGrade,
    HallucinationGrade,
    build_rag_graph,
    invoke_config,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langgraph.checkpoint.memory import InMemorySaver

pytestmark = pytest.mark.unit

_PROMPTS = {
    "grade_documents": "Grade this document.",
    "transform_query": "Rewrite the question.",
    "generate": "Answer using only the context.",
    "hallucination_grader": "Grade groundedness.",
    "answer_grader": "Grade usefulness.",
}


class _FakeDocument:
    def __init__(self, page_content: str) -> None:
        self.page_content = page_content


class _FakeRetriever:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[_FakeDocument]:
        self.queries.append(query)
        return [_FakeDocument(chunk) for chunk in self._chunks]


class _FakeScore:
    def __init__(self, binary_score: str) -> None:
        self.binary_score = binary_score


class _FakeGrader:
    def __init__(self, scores: Iterator[str]) -> None:
        self._scores = scores
        self.call_count = 0

    def invoke(self, _prompt: str) -> _FakeScore:
        self.call_count += 1
        return _FakeScore(next(self._scores))


class _FakeModel:
    """Stands in for the chat model returned by `get_chat_model`. Each of the three
    `with_structured_output` schemas gets its own scripted answer sequence; `invoke` (rewriting
    vs. generation) is distinguished by a marker each node's prompt is known to include."""

    def __init__(
        self,
        *,
        document_relevant: bool,
        hallucination_scores: Iterator[str],
        usefulness_scores: Iterator[str],
        rewritten_question: str = "a rewritten question",
    ) -> None:
        self.document_grader = _FakeGrader(repeat("yes" if document_relevant else "no"))
        self.hallucination_grader = _FakeGrader(hallucination_scores)
        self.answer_grader = _FakeGrader(usefulness_scores)
        self._rewritten_question = rewritten_question
        self.generate_call_count = 0
        self.invoke_calls: list[str] = []

    def with_structured_output(self, schema: object) -> _FakeGrader:
        if schema is DocumentGrade:
            return self.document_grader
        if schema is HallucinationGrade:
            return self.hallucination_grader
        if schema is AnswerUsefulnessGrade:
            return self.answer_grader
        raise AssertionError(f"unexpected schema: {schema}")

    def invoke(self, prompt: str) -> AIMessage:
        self.invoke_calls.append(prompt)
        if "Original question:" in prompt and "Previous rewrite:" in prompt:
            return AIMessage(content=self._rewritten_question)
        self.generate_call_count += 1
        return AIMessage(content=f"generated answer #{self.generate_call_count}")


def _initial_state(question: str) -> dict[str, object]:
    return {
        "question": question,
        "original_question": question,
        "documents": [],
        "documents_sufficient": False,
        "retry_count": 0,
        "answer": "",
        "grounded": False,
        "useful": False,
        "regenerate_count": 0,
    }


def test_invoke_config_sets_thread_id() -> None:
    config = invoke_config("some-thread-id")

    assert config["configurable"]["thread_id"] == "some-thread-id"


def test_first_pass_grounded_and_useful_terminates_immediately(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retriever = _FakeRetriever(["Milvus stores vectors."])
    fake_model = _FakeModel(
        document_relevant=True,
        hallucination_scores=iter(["yes"]),
        usefulness_scores=iter(["yes"]),
    )
    monkeypatch.setattr("self_rag_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer, prompts=_PROMPTS, retriever=fake_retriever
    )
    result = graph.invoke(_initial_state("What is Milvus?"), config=invoke_config("thread-1"))

    assert result["grounded"] is True
    assert result["useful"] is True
    assert result["regenerate_count"] == 0
    assert result["retry_count"] == 0
    assert fake_model.generate_call_count == 1
    assert result["answer"] == "generated answer #1"


def test_ungrounded_generation_triggers_one_regenerate_then_succeeds(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retriever = _FakeRetriever(["some chunk"])
    fake_model = _FakeModel(
        document_relevant=True,
        hallucination_scores=iter(["no", "yes"]),  # first generation ungrounded, second grounded
        usefulness_scores=iter(["yes"]),  # only reached once grounded
    )
    monkeypatch.setattr("self_rag_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer, prompts=_PROMPTS, retriever=fake_retriever
    )
    result = graph.invoke(_initial_state("a tricky question"), config=invoke_config("thread-2"))

    assert result["regenerate_count"] == 1
    assert result["retry_count"] == 0
    assert fake_model.generate_call_count == 2
    assert fake_retriever.queries == ["a tricky question"]  # never re-retrieved
    assert result["answer"] == "generated answer #2"


def test_grounded_but_not_useful_triggers_one_re_retrieve_then_succeeds(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retriever = _FakeRetriever(["some chunk"])
    fake_model = _FakeModel(
        document_relevant=True,
        hallucination_scores=iter(["yes", "yes"]),  # always grounded
        usefulness_scores=iter(["no", "yes"]),  # first answer not useful, second is
        rewritten_question="a more specific question",
    )
    monkeypatch.setattr("self_rag_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer, prompts=_PROMPTS, retriever=fake_retriever
    )
    result = graph.invoke(_initial_state("a vague question"), config=invoke_config("thread-3"))

    assert result["retry_count"] == 1
    assert result["regenerate_count"] == 0
    assert fake_retriever.queries == ["a vague question", "a more specific question"]
    assert result["answer"] == "generated answer #2"


def test_both_caps_reached_terminates_with_best_effort_instead_of_looping_forever(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retriever = _FakeRetriever(["some chunk"])
    fake_model = _FakeModel(
        document_relevant=True,
        hallucination_scores=cycle(["no"]),  # always ungrounded — never reaches usefulness
        usefulness_scores=cycle(["no"]),  # never actually consumed in this scenario
    )
    monkeypatch.setattr("self_rag_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer, prompts=_PROMPTS, retriever=fake_retriever
    )
    result = graph.invoke(
        _initial_state("a persistently ungrounded question"), config=invoke_config("thread-4")
    )

    # Bounded: exactly one initial generate + MAX_REGENERATE regenerates, never more.
    assert fake_model.generate_call_count == MAX_REGENERATE + 1
    assert result["regenerate_count"] == MAX_REGENERATE + 1
    assert result["retry_count"] <= MAX_RETRIES
    assert result["grounded"] is False
    # Best-so-far, not a fabricated special string — see this module's docstring.
    assert result["answer"] == f"generated answer #{MAX_REGENERATE + 1}"
