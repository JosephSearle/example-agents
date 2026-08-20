"""Unit tests for adaptive_rag_agent.graph — pure logic, a fully stubbed model, and a fake
retriever, no network, no real Milvus.

One test per routing branch, each asserting the branch reaches the right terminal answer AND
never touches nodes/dependencies that belong to the *other* branches — e.g. `no_retrieval` must
never call the retriever at all."""

from __future__ import annotations

from typing import TYPE_CHECKING

from adaptive_rag_agent.graph import (
    ComplexityRoute,
    DocumentGrade,
    SubQuestions,
    build_rag_graph,
    invoke_config,
)
from langchain_core.messages import AIMessage
import pytest

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import InMemorySaver

pytestmark = pytest.mark.unit

_PROMPTS = {
    "route": "Classify the question.",
    "generate_direct": "Answer directly.",
    "grade_documents": "Grade this document.",
    "transform_query": "Rewrite the question.",
    "generate": "Answer using only the context.",
    "decompose": "Break the question into sub-questions.",
    "synthesize": "Combine the sub-answers.",
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


class _FakeRoute:
    def __init__(self, complexity: str) -> None:
        self.complexity = complexity


class _FakeGrade:
    def __init__(self, binary_score: str) -> None:
        self.binary_score = binary_score


class _FakeSubQuestions:
    def __init__(self, sub_questions: list[str]) -> None:
        self.sub_questions = sub_questions


class _FakeStructuredCaller:
    def __init__(self, result: object) -> None:
        self.result = result
        self.call_count = 0

    def invoke(self, _prompt: str) -> object:
        self.call_count += 1
        return self.result


class _FakeModel:
    """Stands in for the chat model returned by `get_chat_model`. Each `with_structured_output`
    schema gets its own scripted result; `invoke` (generate_direct / transform_query / generate /
    synthesize) is distinguished by a marker each node's prompt is known to include."""

    def __init__(
        self,
        *,
        complexity: str,
        document_relevant: bool = True,
        sub_questions: list[str] | None = None,
    ) -> None:
        self._route_caller = _FakeStructuredCaller(_FakeRoute(complexity))
        self._grade_caller = _FakeStructuredCaller(_FakeGrade("yes" if document_relevant else "no"))
        self._decompose_caller = _FakeStructuredCaller(
            _FakeSubQuestions(sub_questions or ["sub-question A", "sub-question B"])
        )
        self.invoke_calls: list[str] = []

    def with_structured_output(self, schema: object) -> _FakeStructuredCaller:
        if schema is ComplexityRoute:
            return self._route_caller
        if schema is DocumentGrade:
            return self._grade_caller
        if schema is SubQuestions:
            return self._decompose_caller
        raise AssertionError(f"unexpected schema: {schema}")

    def invoke(self, prompt: str) -> AIMessage:
        self.invoke_calls.append(prompt)
        if "Sub-questions and answers:" in prompt:
            return AIMessage(content="a synthesized answer")
        if "Original question:" in prompt:
            return AIMessage(content="a rewritten question")
        if "Context:" in prompt:
            return AIMessage(content="a grounded sub-answer")
        return AIMessage(content="a direct answer")


def _initial_state(question: str) -> dict[str, object]:
    return {
        "question": question,
        "complexity": "",
        "documents": [],
        "documents_sufficient": False,
        "retry_count": 0,
        "sub_questions": [],
        "sub_answers": [],
        "answer": "",
    }


def test_invoke_config_sets_thread_id() -> None:
    config = invoke_config("some-thread-id")

    assert config["configurable"]["thread_id"] == "some-thread-id"


def test_no_retrieval_branch_never_touches_the_retriever(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retriever = _FakeRetriever(["should never be fetched"])
    fake_model = _FakeModel(complexity="no_retrieval")
    monkeypatch.setattr("adaptive_rag_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer, prompts=_PROMPTS, retriever=fake_retriever
    )
    result = graph.invoke(
        _initial_state("What does idempotent mean?"), config=invoke_config("thread-1")
    )

    assert result["complexity"] == "no_retrieval"
    assert result["answer"] == "a direct answer"
    assert fake_retriever.queries == []
    assert fake_model._decompose_caller.call_count == 0
    assert fake_model._grade_caller.call_count == 0


def test_single_step_branch_uses_corrective_rag_style_retrieval(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retriever = _FakeRetriever(["Milvus stores vectors."])
    fake_model = _FakeModel(complexity="single_step", document_relevant=True)
    monkeypatch.setattr("adaptive_rag_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer, prompts=_PROMPTS, retriever=fake_retriever
    )
    result = graph.invoke(_initial_state("What is Milvus?"), config=invoke_config("thread-2"))

    assert result["complexity"] == "single_step"
    assert result["answer"] == "a grounded sub-answer"  # matches the "Context:" marker
    assert fake_retriever.queries == ["What is Milvus?"]
    assert fake_model._decompose_caller.call_count == 0


def test_multi_step_branch_answers_each_sub_question_and_synthesizes(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retriever = _FakeRetriever(["some chunk"])
    fake_model = _FakeModel(
        complexity="multi_step", sub_questions=["first sub-question", "second sub-question"]
    )
    monkeypatch.setattr("adaptive_rag_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer, prompts=_PROMPTS, retriever=fake_retriever
    )
    result = graph.invoke(_initial_state("Compare X and Y"), config=invoke_config("thread-3"))

    assert result["complexity"] == "multi_step"
    assert result["sub_questions"] == ["first sub-question", "second sub-question"]
    assert result["sub_answers"] == ["a grounded sub-answer", "a grounded sub-answer"]
    assert result["answer"] == "a synthesized answer"
    assert fake_retriever.queries == ["first sub-question", "second sub-question"]
    assert fake_model._grade_caller.call_count == 0  # grade_documents never runs on this branch
