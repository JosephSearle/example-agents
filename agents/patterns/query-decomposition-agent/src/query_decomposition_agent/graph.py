"""Builds the Query Decomposition RAG workflow graph.

Per `docs/patterns/rag/query-decomposition.md`, this splits a multi-part question into
independent sub-questions, answers each one against Milvus, then synthesizes a final answer from
all the sub-answers. Implements the doc's **parallel** decomposition strategy only — sub-questions
are assumed independent, each answered via its own retrieve+generate pass with no dependency on
the others' answers. The doc's **sequential** strategy (each sub-question's retrieval/generation
sees prior sub-Q&A pairs as context, for genuinely dependent multi-hop questions) is an
intentional non-goal here: this repo already demonstrates sequential-with-accumulated-context via
`prompt_chaining_agent`, and dependent-sub-question handling via `orchestrator_workers_agent`; a
third implementation of that same "carry prior context forward" shape wouldn't teach anything new,
so this pattern stays scoped to what's actually novel about it — the decompose/synthesize
bookends around retrieval.

Sub-questions are answered by a single node looping synchronously over `sub_questions`, not via
LangGraph's `Send` API for dynamic fan-out — `map_reduce_agent.graph` already demonstrates that
mechanic in this repo; duplicating it here would blur what this pattern is meant to teach (query
decomposition, not fan-out mechanics).

Reuses `basic_rag_agent`'s Milvus collection (`COLLECTION_NAME`) and embeddings route. Per this
repo's convention, node functions are duplicated rather than imported across RAG patterns — see
`corrective_rag_agent.graph`'s docstring for the same reasoning, which applies here too since
`adaptive-rag-agent` (built after this pattern) duplicates this graph's shape for its multi-step
branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from agents_common import get_chat_model, get_settings
from agents_common.judges import build_production_scorers
from agents_common.prompts import (
    PRODUCTION_ALIAS,
    link_prompts_to_trace as _link_prompts_to_trace,
    load_prompt_version,
    make_prompt_loaders,
    prompt_text,
)
from agents_common.retrieval import Retriever, build_milvus_retriever
from basic_rag_agent.graph import COLLECTION_NAME
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
import structlog

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from mlflow.entities.model_registry import PromptVersion

_logger = structlog.get_logger(__name__)

__all__ = [
    "COLLECTION_NAME",
    "DEFAULT_K",
    "EMBEDDING_GATEWAY_ROUTE",
    "GATEWAY_ROUTE",
    "MAX_SUB_QUESTIONS",
    "PRODUCTION_SCORERS",
    "PROMPT_NAMES",
    "DecompositionState",
    "Retriever",
    "build_rag_graph",
    "invoke_config",
    "link_prompts_to_trace",
    "load_rag_prompt",
    "load_rag_prompt_version",
    "prompt_text",
]

EXPERIMENT_NAME = "query-decomposition-agent"
GATEWAY_ROUTE = "gpt-oss-120b"
EMBEDDING_GATEWAY_ROUTE = "text-embedding"

# Upper bound on how many sub-questions `decompose` may produce — see docs/patterns/rag/
# query-decomposition.md's own "over-decomposing dilutes focus and multiplies calls" warning.
MAX_SUB_QUESTIONS = 4

# Retrieval k — see basic_rag_agent.graph.DEFAULT_K's own note on the k tradeoff.
DEFAULT_K = 4

PROMPT_NAMES = ("decompose", "generate", "synthesize")

PRODUCTION_SCORERS: list[tuple[Any, float]] = build_production_scorers(
    GATEWAY_ROUTE,
    [
        ("grounded_in_context", "query-decomposition-agent-grounded_in_context"),
        ("addresses_original_question", "query-decomposition-agent-addresses_original_question"),
    ],
)

_PROMPT_ALIAS = PRODUCTION_ALIAS

_prompt_loaders = {
    step: make_prompt_loaders(
        f"{EXPERIMENT_NAME}-{step}", experiment_name=EXPERIMENT_NAME, alias=_PROMPT_ALIAS
    )
    for step in PROMPT_NAMES
}


class SubQuestions(BaseModel):
    """Structured output for the decomposition step."""

    sub_questions: list[str] = Field(
        description="2-4 independent, self-contained sub-questions that together cover the "
        "original question. If the original question is already simple/atomic, return it "
        "unchanged as the only sub-question."
    )


class DecompositionState(TypedDict):
    """State threaded through the graph.

    `sub_answers` is index-aligned with `sub_questions` — `sub_answers[i]` answers
    `sub_questions[i]`.
    """

    question: str
    sub_questions: list[str]
    sub_answers: list[str]
    answer: str


def load_rag_prompt_version(step: str, *, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch one step's prompt version from the MLflow prompt registry.

    Args:
        step: One of `PROMPT_NAMES` ("decompose", "generate", "synthesize").
        alias: Prompt registry alias to load. Defaults to the production alias.
    """
    if alias == _PROMPT_ALIAS:
        return _prompt_loaders[step].load_version()
    return load_prompt_version(
        f"{EXPERIMENT_NAME}-{step}", experiment_name=EXPERIMENT_NAME, alias=alias
    )


def load_rag_prompt(step: str, *, alias: str = _PROMPT_ALIAS) -> str:
    """Fetch one step's prompt text from the MLflow prompt registry."""
    return prompt_text(load_rag_prompt_version(step, alias=alias))


def link_prompts_to_trace(prompt_versions: dict[str, PromptVersion], trace_id: str | None) -> None:
    """Link this invocation's prompt version(s) to a trace.

    Only the prompt versions actually supplied (typically whichever steps ran) need to be passed.
    """
    _link_prompts_to_trace(list(prompt_versions.values()), trace_id)


def build_rag_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    embedding_gateway_route: str = EMBEDDING_GATEWAY_ROUTE,
    milvus_uri: str | None = None,
    k: int = DEFAULT_K,
    max_sub_questions: int = MAX_SUB_QUESTIONS,
    prompts: dict[str, str] | None = None,
    retriever: Retriever | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the Query Decomposition RAG workflow.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration)
            or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route called for decomposition, per-sub-question
            generation, and synthesis.
        embedding_gateway_route: MLflow AI Gateway route retrieval embeds sub-questions with.
            Ignored if `retriever` is passed directly.
        milvus_uri: Overrides `Settings.milvus_uri`. Ignored if `retriever` is passed directly.
        k: Number of chunks to retrieve per sub-question.
        max_sub_questions: Overrides `MAX_SUB_QUESTIONS`.
        prompts: Overrides the registry-fetched prompts, keyed by `PROMPT_NAMES`. Defaults to
            `None`, which fetches each step's current `production`-aliased prompt. Pass literal
            strings in tests that need a hermetic build with no MLflow prompt-registry dependency.
        retriever: Overrides the default Milvus-backed retriever. Pass a fake in tests.

    Returns:
        A compiled LangGraph graph, invoked with `{"question": ..., "sub_questions": [],
        "sub_answers": [], "answer": ""}`.
    """
    model = get_chat_model(gateway_route)
    decomposer = model.with_structured_output(SubQuestions)
    active_prompts = (
        prompts if prompts is not None else {step: load_rag_prompt(step) for step in PROMPT_NAMES}
    )
    active_retriever = (
        retriever
        if retriever is not None
        else build_milvus_retriever(
            collection_name=COLLECTION_NAME,
            embedding_gateway_route=embedding_gateway_route,
            milvus_uri=milvus_uri or get_settings().milvus_uri,
            k=k,
        )
    )

    def decompose(state: DecompositionState) -> dict[str, list[str]]:
        result = decomposer.invoke(
            f"{active_prompts['decompose']}\n\nQuestion: {state['question']}"
        )
        sub_questions = result.sub_questions[:max_sub_questions]  # type: ignore[union-attr]
        _logger.info("decomposed", sub_question_count=len(sub_questions))
        return {"sub_questions": sub_questions}

    def answer_sub_questions(state: DecompositionState) -> dict[str, list[str]]:
        sub_answers = []
        for sub_question in state["sub_questions"]:
            documents = active_retriever.invoke(sub_question)
            chunks = [doc.page_content for doc in documents]
            if not chunks:
                sub_answers.append("No relevant context found for this sub-question.")
                continue
            context = "\n\n".join(chunks)
            response = model.invoke(
                f"{active_prompts['generate']}\n\nContext:\n{context}\n\nQuestion: {sub_question}"
            )
            sub_answers.append(str(response.content))
        _logger.info("answered_sub_questions", sub_question_count=len(state["sub_questions"]))
        return {"sub_answers": sub_answers}

    def synthesize(state: DecompositionState) -> dict[str, str]:
        pairs = "\n\n".join(
            f"Q: {q}\nA: {a}"
            for q, a in zip(state["sub_questions"], state["sub_answers"], strict=True)
        )
        response = model.invoke(
            f"{active_prompts['synthesize']}\n\nOriginal question: {state['question']}\n\n"
            f"Sub-questions and answers:\n{pairs}"
        )
        _logger.info("synthesized", sub_answer_count=len(state["sub_answers"]))
        return {"answer": str(response.content)}

    graph = StateGraph(DecompositionState)
    graph.add_node("decompose", decompose)
    graph.add_node("answer_sub_questions", answer_sub_questions)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "answer_sub_questions")
    graph.add_edge("answer_sub_questions", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str) -> dict[str, Any]:
    """Build the `.invoke()` config for a thread."""
    return {"configurable": {"thread_id": thread_id}}
