"""Unit tests for query_decomposition_agent.graph — pure logic, a stubbed model (decompose,
per-sub-question generate, and synthesize all faked), and a fake retriever, no network, no real
Milvus."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
import pytest
from query_decomposition_agent.graph import build_rag_graph, invoke_config

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import InMemorySaver

pytestmark = pytest.mark.unit

_PROMPTS = {
    "decompose": "Break the question into sub-questions.",
    "generate": "Answer using only the context.",
    "synthesize": "Combine the sub-answers.",
}


class _FakeDocument:
    def __init__(self, page_content: str) -> None:
        self.page_content = page_content


class _FakeRetriever:
    """Returns one canned chunk per sub-question (looked up by query), and records every query."""

    def __init__(self, chunks_by_query: dict[str, list[str]]) -> None:
        self._chunks_by_query = chunks_by_query
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[_FakeDocument]:
        self.queries.append(query)
        return [_FakeDocument(chunk) for chunk in self._chunks_by_query.get(query, [])]


class _FakeSubQuestions:
    def __init__(self, sub_questions: list[str]) -> None:
        self.sub_questions = sub_questions


class _FakeDecomposer:
    def __init__(self, sub_questions: list[str]) -> None:
        self._sub_questions = sub_questions
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> _FakeSubQuestions:
        self.calls.append(prompt)
        return _FakeSubQuestions(self._sub_questions)


class _FakeModel:
    """Stands in for the chat model returned by `get_chat_model` — serves decomposition (via
    `with_structured_output`), per-sub-question generation, and synthesis, distinguishing the
    latter two by a marker each node's prompt is known to include."""

    def __init__(
        self, *, sub_questions: list[str], sub_answers: dict[str, str], synthesized: str
    ) -> None:
        self._decomposer = _FakeDecomposer(sub_questions)
        self._sub_answers = sub_answers
        self._synthesized = synthesized
        self.invoke_calls: list[str] = []

    def with_structured_output(self, _schema: object) -> _FakeDecomposer:
        return self._decomposer

    def invoke(self, prompt: str) -> AIMessage:
        self.invoke_calls.append(prompt)
        if "Sub-questions and answers:" in prompt:
            return AIMessage(content=self._synthesized)
        for sub_question, answer in self._sub_answers.items():
            if sub_question in prompt:
                return AIMessage(content=answer)
        raise AssertionError(f"no fake answer configured for prompt: {prompt}")


def _initial_state(question: str) -> dict[str, object]:
    return {"question": question, "sub_questions": [], "sub_answers": [], "answer": ""}


def test_invoke_config_sets_thread_id() -> None:
    config = invoke_config("some-thread-id")

    assert config["configurable"]["thread_id"] == "some-thread-id"


def test_sub_answers_stay_index_aligned_with_sub_questions_and_synthesis_sees_all_pairs(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    sub_questions = ["What tier does react-agent use?", "What tier does swarm-agent use?"]
    fake_retriever = _FakeRetriever(
        {
            "What tier does react-agent use?": ["react-agent uses Tier 1."],
            "What tier does swarm-agent use?": ["swarm-agent uses Tier 2."],
        }
    )
    fake_model = _FakeModel(
        sub_questions=sub_questions,
        sub_answers={
            "What tier does react-agent use?": "react-agent uses Tier 1.",
            "What tier does swarm-agent use?": "swarm-agent uses Tier 2.",
        },
        synthesized="react-agent uses Tier 1 and swarm-agent uses Tier 2.",
    )
    monkeypatch.setattr("query_decomposition_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer, prompts=_PROMPTS, retriever=fake_retriever
    )
    result = graph.invoke(
        _initial_state("What tiers do react-agent and swarm-agent use?"),
        config=invoke_config("thread-1"),
    )

    assert result["sub_questions"] == sub_questions
    assert result["sub_answers"] == ["react-agent uses Tier 1.", "swarm-agent uses Tier 2."]
    assert result["answer"] == "react-agent uses Tier 1 and swarm-agent uses Tier 2."
    assert fake_retriever.queries == sub_questions
    synthesis_prompt = next(p for p in fake_model.invoke_calls if "Sub-questions and answers:" in p)
    assert "react-agent uses Tier 1." in synthesis_prompt
    assert "swarm-agent uses Tier 2." in synthesis_prompt


def test_sub_question_with_empty_retrieval_gets_an_honest_placeholder_answer(
    in_memory_checkpointer: InMemorySaver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_retriever = _FakeRetriever({})  # every query returns no chunks
    fake_model = _FakeModel(
        sub_questions=["An unanswerable sub-question?"],
        sub_answers={},
        synthesized="final synthesis",
    )
    monkeypatch.setattr("query_decomposition_agent.graph.get_chat_model", lambda _route: fake_model)

    graph = build_rag_graph(
        checkpointer=in_memory_checkpointer, prompts=_PROMPTS, retriever=fake_retriever
    )
    result = graph.invoke(_initial_state("some question"), config=invoke_config("thread-2"))

    assert result["sub_answers"] == ["No relevant context found for this sub-question."]
    assert result["answer"] == "final synthesis"
