"""Builds the Retrieve & Rerank RAG workflow graph.

Per `docs/patterns/rag/retrieve-rerank.md`, this sits in the same "2-Step RAG" tier as Basic RAG
(see `docs/patterns/rag/adoption-strategy.md`) — it's a precision fix layered on top of retrieval,
not a new control-flow architecture. The doc itself has no LangGraph at all (`retriever.invoke` ->
reranker -> `llm.invoke`, a linear composition) but this repo implements every pattern as a
`StateGraph`, even trivial linear ones (see `basic_rag_agent.graph`'s own "workflow, not agent"
framing) — for consistency across patterns, and because checkpointing/tracing per node is worth
having even when there's no branching.

Retrieval fetches a wide candidate set (`DEFAULT_CANDIDATE_K`, over-fetch on purpose) and a
cross-encoder reranker narrows it to the few chunks (`DEFAULT_TOP_N`) actually worth putting in
the generation prompt — see `retrieve-rerank.md`'s own tuning note: reranking cannot recover from
a bad first-pass retrieval (if the right chunk isn't in the candidate set, no rerank fixes it), and
too-small a candidate set defeats the point of reranking at all.

Reuses `basic_rag_agent`'s Milvus collection (`COLLECTION_NAME`) and embeddings route — this
pattern is about narrowing what's already retrievable, not a different corpus. The reranker itself
is a *separate* self-hosted model, reached directly over HTTP rather than through the MLflow AI
Gateway — see `agents_common.models.get_reranker`'s docstring for why a cross-encoder reranker
doesn't fit the gateway's chat/embeddings-shaped provisioning at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from agents_common import get_chat_model, get_reranker, get_settings
from agents_common.judges import build_production_scorers
from agents_common.prompts import (
    PRODUCTION_ALIAS,
    link_prompts_to_trace,
    load_prompt_version,
    make_prompt_loaders,
    prompt_text,
)
from agents_common.retrieval import NO_CONTEXT_ANSWER, Retriever, build_milvus_retriever
from basic_rag_agent.graph import COLLECTION_NAME
from langgraph.graph import END, START, StateGraph
import structlog

if TYPE_CHECKING:
    from agents_common.models import RerankResult
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from mlflow.entities.model_registry import PromptVersion

_logger = structlog.get_logger(__name__)

__all__ = [
    "COLLECTION_NAME",
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_TOP_N",
    "EMBEDDING_GATEWAY_ROUTE",
    "GATEWAY_ROUTE",
    "NO_CONTEXT_ANSWER",
    "PRODUCTION_SCORERS",
    "RerankRagState",
    "Reranker",
    "Retriever",
    "build_rag_graph",
    "invoke_config",
    "link_prompt_to_trace",
    "load_rag_prompt",
    "load_rag_prompt_version",
    "prompt_text",
]

EXPERIMENT_NAME = "retrieve-rerank-agent"

# Same chat/embeddings routes basic-rag-agent uses — no new gateway provisioning needed for this
# pattern, only the reranker (see agents_common.models.get_reranker) is new infrastructure.
GATEWAY_ROUTE = "gpt-oss-120b"
EMBEDDING_GATEWAY_ROUTE = "text-embedding"

# Over-fetch on retrieval so the reranker has a real candidate pool to narrow down — see this
# module's docstring on why too-small a candidate set defeats reranking's purpose.
DEFAULT_CANDIDATE_K = 20

# How many reranked chunks actually reach the generation prompt.
DEFAULT_TOP_N = 5

PRODUCTION_SCORERS: list[tuple[Any, float]] = build_production_scorers(
    GATEWAY_ROUTE, [("grounded_in_context", "retrieve-rerank-agent-grounded_in_context")]
)

_PROMPT_ALIAS = PRODUCTION_ALIAS

_prompt_loaders = make_prompt_loaders(EXPERIMENT_NAME, experiment_name=EXPERIMENT_NAME)


class Reranker(Protocol):
    """The shape `build_rag_graph`'s `reranker` override needs to satisfy.

    A real `agents_common.models.get_reranker()` client satisfies this automatically; tests pass
    a lightweight fake instead.
    """

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[RerankResult]:
        """Return the top `top_n` `documents` (by index), sorted by descending relevance."""
        ...


class RerankRagState(TypedDict):
    """State threaded through the graph.

    `candidate_chunks` is the wide pre-rerank retrieval; `reranked_chunks` is what actually
    reaches `generate`. Kept as separate fields (rather than overwriting one list) so a trace
    reviewer can see both the retriever's and the reranker's contribution independently.
    """

    question: str
    candidate_chunks: list[str]
    reranked_chunks: list[str]
    answer: str


def load_rag_prompt_version(*, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch this agent's generation prompt version from the MLflow prompt registry."""
    if alias == _PROMPT_ALIAS:
        return _prompt_loaders.load_version()
    return load_prompt_version(EXPERIMENT_NAME, experiment_name=EXPERIMENT_NAME, alias=alias)


def load_rag_prompt(*, alias: str = _PROMPT_ALIAS) -> str:
    """Fetch this agent's generation prompt text from the MLflow prompt registry."""
    return prompt_text(load_rag_prompt_version(alias=alias))


def link_prompt_to_trace(prompt_version: PromptVersion, trace_id: str | None) -> None:
    """Link the generation prompt version to a trace so the MLflow UI's trace view shows it."""
    link_prompts_to_trace([prompt_version], trace_id)


def build_rag_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    embedding_gateway_route: str = EMBEDDING_GATEWAY_ROUTE,
    milvus_uri: str | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    top_n: int = DEFAULT_TOP_N,
    rag_prompt: str | None = None,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the Retrieve & Rerank RAG workflow.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration)
            or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route the generation step calls.
        embedding_gateway_route: MLflow AI Gateway route retrieval embeds queries with. Ignored
            if `retriever` is passed directly.
        milvus_uri: Overrides `Settings.milvus_uri`. Ignored if `retriever` is passed directly.
        candidate_k: Number of chunks the retriever fetches before reranking.
        top_n: Number of reranked chunks passed to generation.
        rag_prompt: Overrides the registry-fetched generation prompt.
        retriever: Overrides the default Milvus-backed retriever. Pass a fake in tests.
        reranker: Overrides the default HTTP reranker client (see `agents_common.models.get_reranker`).
            Pass a fake in tests that need a hermetic build with no real reranker HTTP call.

    Returns:
        A compiled LangGraph graph, invoked with `{"question": ..., "candidate_chunks": [],
        "reranked_chunks": [], "answer": ""}`.
    """
    model = get_chat_model(gateway_route)
    prompt = rag_prompt if rag_prompt is not None else load_rag_prompt()
    active_retriever = (
        retriever
        if retriever is not None
        else build_milvus_retriever(
            collection_name=COLLECTION_NAME,
            embedding_gateway_route=embedding_gateway_route,
            milvus_uri=milvus_uri or get_settings().milvus_uri,
            k=candidate_k,
        )
    )
    active_reranker = reranker if reranker is not None else get_reranker()

    def retrieve(state: RerankRagState) -> dict[str, list[str]]:
        documents = active_retriever.invoke(state["question"])
        chunks = [doc.page_content for doc in documents]
        _logger.info("retrieved", chunk_count=len(chunks))
        return {"candidate_chunks": chunks}

    def rerank(state: RerankRagState) -> dict[str, list[str]]:
        candidates = state["candidate_chunks"]
        if not candidates:
            _logger.info("reranked", chunk_count=0)
            return {"reranked_chunks": []}
        results = active_reranker.rerank(state["question"], candidates, top_n=top_n)
        reranked = [candidates[result["index"]] for result in results]
        _logger.info("reranked", candidate_count=len(candidates), chunk_count=len(reranked))
        return {"reranked_chunks": reranked}

    def generate(state: RerankRagState) -> dict[str, str]:
        if not state["reranked_chunks"]:
            _logger.info("generated", grounded=False)
            return {"answer": NO_CONTEXT_ANSWER}

        context = "\n\n".join(state["reranked_chunks"])
        response = model.invoke(f"{prompt}\n\nContext:\n{context}\n\nQuestion: {state['question']}")
        _logger.info("generated", grounded=True, chunk_count=len(state["reranked_chunks"]))
        return {"answer": str(response.content)}

    graph = StateGraph(RerankRagState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str) -> dict[str, Any]:
    """Build the `.invoke()` config for a thread."""
    return {"configurable": {"thread_id": thread_id}}
