"""Builds the Corrective RAG (CRAG) workflow graph.

Per `docs/patterns/rag/corrective-rag-crag.md`, this grades retrieved documents before trusting
them: each retrieved chunk gets a binary relevance grade from an LLM judge, and if none of them
are relevant, the question is rewritten and retrieval is retried rather than generating from bad
context. Implements the doc's **Beginner** (binary gate) design only — the **Advanced** 3-tier
confidence router (correct/incorrect/ambiguous, with strip-level re-grading) is an intentional
non-goal here, per `docs/patterns/rag/adoption-strategy.md`'s "adopt the diagnosed failure, don't
stack speculative sophistication."

**Deviation from the doc**: the reference design falls back to an external web search (Tavily)
when retrieved documents are graded insufficient. This repo has a strict one-credential
(`MLFLOW_TRACKING_TOKEN`) rule with no room for a second, unrelated API key, so that branch is
replaced here with "rewrite the question and retry retrieval against the same Milvus collection"
(`transform_query` loops back into `retrieve`) — corrective, but scoped to this repo's own corpus
rather than the open web. Looping back into `retrieve` (rather than the doc's one-shot
rewrite-then-web-search-then-generate) risks looping forever if the rewritten query is still
graded poorly, so a `retry_count` cap (`MAX_RETRIES`) forces the graph to `generate` anyway once
exhausted, from whatever was retrieved — an honest best-effort rather than an infinite loop.

Reuses `basic_rag_agent`'s Milvus collection (`COLLECTION_NAME`) and embeddings route. Per this
repo's convention (see `basic_rag_agent.graph`'s own precedent), node *functions* are duplicated
rather than imported across RAG patterns — this graph is meant to be readable standalone, and
`self-rag-agent`/`adaptive-rag-agent` (built after this pattern) duplicate this shape rather than
importing it, so editing this file's grading prompt doesn't silently change their behavior.
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
    "NO_CONTEXT_ANSWER",
    "PRODUCTION_SCORERS",
    "PROMPT_NAMES",
    "CragState",
    "Retriever",
    "build_rag_graph",
    "invoke_config",
    "link_prompts_to_trace",
    "load_rag_prompt",
    "load_rag_prompt_version",
    "prompt_text",
]

EXPERIMENT_NAME = "corrective-rag-agent"
GATEWAY_ROUTE = "gpt-oss-120b"
EMBEDDING_GATEWAY_ROUTE = "text-embedding"

# Retrieval-retry cap — the infinite-loop guard described in this module's docstring. Once hit,
# `generate` runs from whatever was retrieved instead of looping again.
MAX_RETRIES = 2

# Retrieval k — see basic-rag.md's own note on the k tradeoff (too low misses context, too high
# floods the prompt with noise). Same convention as basic_rag_agent.graph.DEFAULT_K.
DEFAULT_K = 4

PROMPT_NAMES = ("grade_documents", "transform_query", "generate")

PRODUCTION_SCORERS: list[tuple[Any, float]] = build_production_scorers(
    GATEWAY_ROUTE, [("grounded_in_context", "corrective-rag-agent-grounded_in_context")]
)

# One `PromptLoaders` per step, keyed the same way each step's registry name is built
# (f"{EXPERIMENT_NAME}-{step}") — see `load_rag_prompt_version`/`load_rag_prompt` below.
_prompt_loaders = {
    step: make_prompt_loaders(f"{EXPERIMENT_NAME}-{step}", experiment_name=EXPERIMENT_NAME)
    for step in PROMPT_NAMES
}


class DocumentGrade(BaseModel):
    """Structured output for the per-document relevance grader."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to answering the question, else 'no'."
    )


class CragState(TypedDict):
    """State threaded through the graph.

    `original_question` is preserved untouched through any `transform_query` rewrites, so
    `generate`'s prompt and any downstream eval always has the true original ask available.
    `documents_sufficient` is written by `grade_documents` and read by the `decide_to_generate`
    conditional edge — named for what it now means (post web-search-removal) rather than the
    doc's own `web_search_needed` field name.
    """

    question: str
    original_question: str
    documents: list[str]
    documents_sufficient: bool
    retry_count: int
    answer: str


def load_rag_prompt_version(step: str, *, alias: str = PRODUCTION_ALIAS) -> PromptVersion:
    """Fetch one step's prompt version from the MLflow prompt registry.

    Args:
        step: One of `PROMPT_NAMES` ("grade_documents", "transform_query", "generate").
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
    max_retries: int = MAX_RETRIES,
    prompts: dict[str, str] | None = None,
    retriever: Retriever | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the Corrective RAG workflow.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration)
            or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route called for grading, rewriting, and generation.
        embedding_gateway_route: MLflow AI Gateway route retrieval embeds queries with. Ignored
            if `retriever` is passed directly.
        milvus_uri: Overrides `Settings.milvus_uri`. Ignored if `retriever` is passed directly.
        k: Number of chunks to retrieve per attempt.
        max_retries: Overrides `MAX_RETRIES` — the retrieval-retry cap.
        prompts: Overrides the registry-fetched prompts, keyed by `PROMPT_NAMES`. Defaults to
            `None`, which fetches each step's current `production`-aliased prompt. Pass literal
            strings in tests that need a hermetic build with no MLflow prompt-registry dependency.
        retriever: Overrides the default Milvus-backed retriever. Pass a fake in tests.

    Returns:
        A compiled LangGraph graph, invoked with `{"question": ..., "original_question": ...,
        "documents": [], "documents_sufficient": False, "retry_count": 0, "answer": ""}`.
    """
    model = get_chat_model(gateway_route)
    grader = model.with_structured_output(DocumentGrade)
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

    def retrieve(state: CragState) -> dict[str, list[str]]:
        documents = active_retriever.invoke(state["question"])
        chunks = [doc.page_content for doc in documents]
        _logger.info("retrieved", chunk_count=len(chunks), retry_count=state["retry_count"])
        return {"documents": chunks}

    def grade_documents(state: CragState) -> dict[str, list[str] | bool]:
        relevant = []
        for document in state["documents"]:
            grade = grader.invoke(
                f"{active_prompts['grade_documents']}\n\n"
                f"Question: {state['question']}\n\nDocument: {document}"
            )
            if grade.binary_score == "yes":  # type: ignore[union-attr]
                relevant.append(document)
        sufficient = len(relevant) > 0
        _logger.info(
            "graded_documents",
            candidate_count=len(state["documents"]),
            relevant_count=len(relevant),
            sufficient=sufficient,
        )
        return {"documents": relevant, "documents_sufficient": sufficient}

    def decide_to_generate(state: CragState) -> str:
        if state["documents_sufficient"]:
            return "generate"
        if state["retry_count"] < max_retries:
            return "transform_query"
        _logger.warning(
            "corrective_rag_retry_cap_reached",
            retry_count=state["retry_count"],
            max_retries=max_retries,
        )
        return "generate"

    def transform_query(state: CragState) -> dict[str, str | int]:
        response = model.invoke(
            f"{active_prompts['transform_query']}\n\nOriginal question: {state['original_question']}\n"
            f"Previous rewrite: {state['question']}"
        )
        rewritten = str(response.content)
        _logger.info("transformed_query", retry_count=state["retry_count"] + 1)
        return {"question": rewritten, "retry_count": state["retry_count"] + 1}

    def generate(state: CragState) -> dict[str, str]:
        if not state["documents"]:
            _logger.info("generated", grounded=False)
            return {"answer": NO_CONTEXT_ANSWER}

        context = "\n\n".join(state["documents"])
        response = model.invoke(
            f"{active_prompts['generate']}\n\nContext:\n{context}\n\n"
            f"Question: {state['original_question']}"
        )
        _logger.info("generated", grounded=True, chunk_count=len(state["documents"]))
        return {"answer": str(response.content)}

    graph = StateGraph(CragState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("transform_query", transform_query)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str) -> dict[str, Any]:
    """Build the `.invoke()` config for a thread."""
    return {"configurable": {"thread_id": thread_id}}
