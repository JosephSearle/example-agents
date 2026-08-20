"""Builds the Adaptive RAG workflow graph.

Per `docs/patterns/rag/adaptive-rag.md`, this routes each question by complexity *before* any
retrieval happens (a conditional entry point, not a conditional edge after some fixed first node)
to one of three branches:

- `no_retrieval` -> answer directly from the model's own knowledge, no retrieval at all.
- `single_step` -> `corrective_rag_agent`'s whole graph (retrieve, grade, retry-or-generate).
- `multi_step` -> `query_decomposition_agent`'s whole graph (decompose, answer sub-questions,
  synthesize).

**Deviation from the doc, and the one place this plan departs from "follow the Beginner-then-
Advanced structure every other pattern in this plan follows"**: the doc's Beginner design is a
*binary* router (`vectorstore` vs `web_search`) feeding a shared graph. With web search removed
repo-wide (see `corrective_rag_agent.graph`'s docstring), a binary router missing one of its two
branches is degenerate — it would always pick `vectorstore`, i.e. just be `corrective-rag-agent`
with extra router overhead, demonstrating nothing new. This implements the **Advanced** 3-way
complexity router directly instead, since that's the version that actually demonstrates a routing
*decision*, not just "always retrieve."

**Multi-step branch chose Query Decomposition over Self-RAG** (the doc offers either): stacking
Self-RAG's own cyclic retry machinery *underneath* an already-branching router risks a
combinatorially confusing graph for a reference implementation whose whole point is demonstrating
the routing decision cleanly. Query Decomposition's linear shape keeps this graph's complexity
focused on the one thing it's meant to teach.

Per this repo's convention (see `corrective_rag_agent.graph`'s docstring), node functions are
**duplicated**, not imported, from `corrective_rag_agent`/`query_decomposition_agent` — this is
the pattern with the most duplicated code in the whole RAG series, and that's an accepted
tradeoff: a reader should be able to understand what `single_step`/`multi_step` actually do
without opening two other packages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

from agents_common import get_chat_model, get_settings
from agents_common.judges import build_production_scorers
from agents_common.prompts import (
    PRODUCTION_ALIAS,
    link_prompts_to_trace as _link_prompts_to_trace,
    make_prompt_loaders,
    prompt_text,
)
from agents_common.retrieval import NO_CONTEXT_ANSWER, Retriever, build_milvus_retriever
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
    "MAX_RETRIES",
    "MAX_SUB_QUESTIONS",
    "NO_CONTEXT_ANSWER",
    "PRODUCTION_SCORERS",
    "PROMPT_NAMES",
    "AdaptiveRagState",
    "Retriever",
    "build_rag_graph",
    "invoke_config",
    "link_prompts_to_trace",
    "load_rag_prompt",
    "load_rag_prompt_version",
    "prompt_text",
]

EXPERIMENT_NAME = "adaptive-rag-agent"
GATEWAY_ROUTE = "gpt-oss-120b"
EMBEDDING_GATEWAY_ROUTE = "text-embedding"

MAX_RETRIES = 2
MAX_SUB_QUESTIONS = 4

# Retrieval k — see basic-rag.md's own note on the k tradeoff (too low misses context, too high
# floods the prompt with noise). Same convention as basic_rag_agent.graph.DEFAULT_K.
DEFAULT_K = 4

PROMPT_NAMES = (
    "route",
    "generate_direct",
    "grade_documents",
    "transform_query",
    "generate",
    "decompose",
    "synthesize",
)

PRODUCTION_SCORERS: list[tuple[Any, float]] = build_production_scorers(
    GATEWAY_ROUTE,
    [
        ("grounded_in_context", "adaptive-rag-agent-grounded_in_context"),
        ("routed_appropriately", "adaptive-rag-agent-routed_appropriately"),
    ],
)

# One `PromptLoaders` per step, keyed the same way each step's registry name is built
# (f"{EXPERIMENT_NAME}-{step}") — see `load_rag_prompt_version`/`load_rag_prompt` below.
_prompt_loaders = {
    step: make_prompt_loaders(f"{EXPERIMENT_NAME}-{step}", experiment_name=EXPERIMENT_NAME)
    for step in PROMPT_NAMES
}


class ComplexityRoute(BaseModel):
    """Structured output for the complexity router — the conditional entry point's classifier."""

    complexity: Literal["no_retrieval", "single_step", "multi_step"] = Field(
        description=(
            "'no_retrieval' if the model can answer confidently from general knowledge with no "
            "lookup needed; 'single_step' if it needs a single retrieval pass against this "
            "repo's own docs; 'multi_step' if it has multiple independent parts each needing "
            "their own retrieval."
        )
    )


class DocumentGrade(BaseModel):
    """Structured output for the per-document relevance grader (single_step branch).

    Same shape as `corrective_rag_agent.graph.DocumentGrade`.
    """

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to answering the question, else 'no'."
    )


class SubQuestions(BaseModel):
    """Structured output for the decomposition step (multi_step branch).

    Same shape as `query_decomposition_agent.graph.SubQuestions`.
    """

    sub_questions: list[str] = Field(
        description="2-4 independent, self-contained sub-questions that together cover the "
        "original question."
    )


class AdaptiveRagState(TypedDict):
    """State threaded through the graph — a flat superset covering all three branches.

    Unused fields on any given branch just stay at their initial empty/zero value, same
    convention as `basic_rag_agent.graph.RagState` not needing every field touched by every node.
    """

    question: str
    complexity: str
    documents: list[str]
    documents_sufficient: bool
    retry_count: int
    sub_questions: list[str]
    sub_answers: list[str]
    answer: str


def load_rag_prompt_version(step: str, *, alias: str = PRODUCTION_ALIAS) -> PromptVersion:
    """Fetch one step's prompt version from the MLflow prompt registry.

    Args:
        step: One of `PROMPT_NAMES`.
        alias: Prompt registry alias to load. Defaults to the production alias; only builds a
            fresh loader when a non-default alias is requested.
    """
    if alias == PRODUCTION_ALIAS:
        return _prompt_loaders[step].load_version()
    return make_prompt_loaders(
        f"{EXPERIMENT_NAME}-{step}", experiment_name=EXPERIMENT_NAME, alias=alias
    ).load_version()


def load_rag_prompt(step: str, *, alias: str = PRODUCTION_ALIAS) -> str:
    """Fetch one step's prompt text from the MLflow prompt registry."""
    return prompt_text(load_rag_prompt_version(step, alias=alias))


def link_prompts_to_trace(prompt_versions: dict[str, PromptVersion], trace_id: str | None) -> None:
    """Link this invocation's prompt version(s) to a trace.

    Only the prompt versions actually supplied (typically whichever branch ran) need to be passed.
    """
    _link_prompts_to_trace(list(prompt_versions.values()), trace_id)


def _build_single_step_nodes(
    *,
    model: Any,
    document_grader: Any,
    active_prompts: dict[str, str],
    active_retriever: Retriever,
    max_retries: int,
) -> tuple[Any, Any, Any, Any, Any]:
    """Build the single_step branch's node/edge callables.

    Duplicated from `corrective_rag_agent.graph` (retrieve, grade, retry-or-generate), pulled
    into a helper to keep `build_rag_graph`'s own statement count reasonable.
    """

    def retrieve(state: AdaptiveRagState) -> dict[str, list[str]]:
        documents = active_retriever.invoke(state["question"])
        chunks = [doc.page_content for doc in documents]
        _logger.info("retrieved", chunk_count=len(chunks), retry_count=state["retry_count"])
        return {"documents": chunks}

    def grade_documents(state: AdaptiveRagState) -> dict[str, list[str] | bool]:
        relevant = []
        for document in state["documents"]:
            grade = document_grader.invoke(
                f"{active_prompts['grade_documents']}\n\n"
                f"Question: {state['question']}\n\nDocument: {document}"
            )
            if grade.binary_score == "yes":
                relevant.append(document)
        sufficient = len(relevant) > 0
        _logger.info(
            "graded_documents",
            candidate_count=len(state["documents"]),
            relevant_count=len(relevant),
            sufficient=sufficient,
        )
        return {"documents": relevant, "documents_sufficient": sufficient}

    def decide_to_generate(state: AdaptiveRagState) -> str:
        if state["documents_sufficient"]:
            return "generate"
        if state["retry_count"] < max_retries:
            return "transform_query"
        _logger.warning(
            "adaptive_rag_retrieval_cap_reached",
            retry_count=state["retry_count"],
            max_retries=max_retries,
        )
        return "generate"

    def transform_query(state: AdaptiveRagState) -> dict[str, str | int]:
        response = model.invoke(
            f"{active_prompts['transform_query']}\n\nOriginal question: {state['question']}"
        )
        rewritten = str(response.content)
        _logger.info("transformed_query", retry_count=state["retry_count"] + 1)
        return {"question": rewritten, "retry_count": state["retry_count"] + 1}

    def generate(state: AdaptiveRagState) -> dict[str, str]:
        if not state["documents"]:
            _logger.info("generated", grounded=False)
            return {"answer": NO_CONTEXT_ANSWER}

        context = "\n\n".join(state["documents"])
        response = model.invoke(
            f"{active_prompts['generate']}\n\nContext:\n{context}\n\nQuestion: {state['question']}"
        )
        _logger.info("generated", chunk_count=len(state["documents"]))
        return {"answer": str(response.content)}

    return retrieve, grade_documents, decide_to_generate, transform_query, generate


def _build_multi_step_nodes(
    *, model: Any, decomposer: Any, active_prompts: dict[str, str], active_retriever: Retriever
) -> tuple[Any, Any, Any]:
    """Build the multi_step branch's node callables.

    Duplicated from `query_decomposition_agent.graph` (decompose, answer sub-questions,
    synthesize), pulled into a helper to keep `build_rag_graph`'s own statement count reasonable.
    """

    def decompose(state: AdaptiveRagState) -> dict[str, list[str]]:
        result = decomposer.invoke(
            f"{active_prompts['decompose']}\n\nQuestion: {state['question']}"
        )
        sub_questions = result.sub_questions[:MAX_SUB_QUESTIONS]
        _logger.info("decomposed", sub_question_count=len(sub_questions))
        return {"sub_questions": sub_questions}

    def answer_sub_questions(state: AdaptiveRagState) -> dict[str, list[str]]:
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

    def synthesize(state: AdaptiveRagState) -> dict[str, str]:
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

    return decompose, answer_sub_questions, synthesize


def build_rag_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    embedding_gateway_route: str = EMBEDDING_GATEWAY_ROUTE,
    milvus_uri: str | None = None,
    k: int = DEFAULT_K,
    max_retries: int = MAX_RETRIES,
    prompts: dict[str, str] | None = None,
    retriever: Retriever | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the Adaptive RAG workflow.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration)
            or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route called for every step across all three branches.
        embedding_gateway_route: MLflow AI Gateway route retrieval embeds queries with. Ignored
            if `retriever` is passed directly.
        milvus_uri: Overrides `Settings.milvus_uri`. Ignored if `retriever` is passed directly.
        k: Number of chunks to retrieve per attempt (single_step and multi_step branches).
        max_retries: Overrides `MAX_RETRIES` — the single_step branch's retrieval-retry cap.
        prompts: Overrides the registry-fetched prompts, keyed by `PROMPT_NAMES`. Defaults to
            `None`, which fetches each step's current `production`-aliased prompt. Pass literal
            strings in tests that need a hermetic build with no MLflow prompt-registry dependency.
        retriever: Overrides the default Milvus-backed retriever. Pass a fake in tests.

    Returns:
        A compiled LangGraph graph, invoked with `{"question": ..., "complexity": "",
        "documents": [], "documents_sufficient": False, "retry_count": 0, "sub_questions": [],
        "sub_answers": [], "answer": ""}`.
    """
    model = get_chat_model(gateway_route)
    router = model.with_structured_output(ComplexityRoute)
    document_grader = model.with_structured_output(DocumentGrade)
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

    def route_by_complexity(state: AdaptiveRagState) -> dict[str, str]:
        result = router.invoke(f"{active_prompts['route']}\n\nQuestion: {state['question']}")
        _logger.info("routed", complexity=result.complexity)  # type: ignore[union-attr]
        return {"complexity": result.complexity}  # type: ignore[union-attr]

    def route_from_complexity(state: AdaptiveRagState) -> str:
        return state["complexity"]

    def generate_direct(state: AdaptiveRagState) -> dict[str, str]:
        response = model.invoke(
            f"{active_prompts['generate_direct']}\n\nQuestion: {state['question']}"
        )
        _logger.info("generated_direct")
        return {"answer": str(response.content)}

    retrieve, grade_documents, decide_to_generate, transform_query, generate = (
        _build_single_step_nodes(
            model=model,
            document_grader=document_grader,
            active_prompts=active_prompts,
            active_retriever=active_retriever,
            max_retries=max_retries,
        )
    )
    decompose, answer_sub_questions, synthesize = _build_multi_step_nodes(
        model=model,
        decomposer=decomposer,
        active_prompts=active_prompts,
        active_retriever=active_retriever,
    )

    graph = StateGraph(AdaptiveRagState)
    graph.add_node("route_by_complexity", route_by_complexity)
    graph.add_node("generate_direct", generate_direct)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("transform_query", transform_query)
    graph.add_node("generate", generate)
    graph.add_node("decompose", decompose)
    graph.add_node("answer_sub_questions", answer_sub_questions)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "route_by_complexity")
    graph.add_conditional_edges(
        "route_by_complexity",
        route_from_complexity,
        {
            "no_retrieval": "generate_direct",
            "single_step": "retrieve",
            "multi_step": "decompose",
        },
    )
    graph.add_edge("generate_direct", END)

    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("generate", END)

    graph.add_edge("decompose", "answer_sub_questions")
    graph.add_edge("answer_sub_questions", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str) -> dict[str, Any]:
    """Build the `.invoke()` config for a thread."""
    return {"configurable": {"thread_id": thread_id}}
