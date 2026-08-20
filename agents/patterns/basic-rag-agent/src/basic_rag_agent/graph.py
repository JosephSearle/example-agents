"""Builds the basic RAG (retrieve-then-generate) workflow graph.

This is a workflow, not an agent, per docs/patterns/rag/basic-rag.md's own framing: retrieval
always happens before generation, on every call — there's no LLM decision about *whether* to
retrieve (that's adaptive-rag.md, not implemented in this repo) and no re-ranking, query
rewriting, or iterative retrieval (retrieve-rerank.md / corrective-rag-crag.md / self-rag.md,
also not implemented). Built with a raw LangGraph `StateGraph` rather than
`langchain.agents.create_agent` (contrast with `react_agent.graph`), since there's no tool loop
here for `create_agent` to compile — same "workflow, not agent" framing
`routing_agent.graph`/`prompt_chaining_agent.graph` use for their own module docstrings.

basic-rag.md itself splits into two concerns: **indexing** (offline, one-time — populating a
Milvus collection) and **retrieval + generation** (this module, the live per-query half). Its own
links for the indexing half (`../milvus/collection-creation`, `../milvus/setup`) are broken — those
docs don't exist in this repo. `packages/milvus/scripts/provision_collections.py` (run via `make
provision-milvus-collections`, chained into `make up`) is this repo's actual answer: it seeds the
`basic_rag_agent` Milvus collection this module queries from
`packages/milvus/collections/basic-rag-agent.jsonl` (chunks of this repo's own
`docs/patterns/{agent,rag}/*.md`).

Per the doc's own called-out failure mode ("Silent failure on empty retrieval"), `generate` below
explicitly branches on an empty `retrieved_chunks` list rather than always forwarding whatever
`k` chunks came back — an empty context block handed to the model risks a confident hallucination
instead of an honest "I don't know."
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from agents_common import get_chat_model, get_embeddings, get_settings
from agents_common.judges import build_production_scorers
from agents_common.prompts import (
    PRODUCTION_ALIAS,
    link_prompts_to_trace,
    load_prompt_version,
    prompt_text,
)
from langchain_milvus import Milvus
from langgraph.graph import END, START, StateGraph
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
    "NO_CONTEXT_ANSWER",
    "PRODUCTION_SCORERS",
    "RagState",
    "Retriever",
    "build_rag_graph",
    "invoke_config",
    "link_prompt_to_trace",
    "load_rag_prompt",
    "load_rag_prompt_version",
    "prompt_text",
]

# This agent's own MLflow experiment — see agents_common.observability.configure_mlflow.
EXPERIMENT_NAME = "basic-rag-agent"

# The MLflow AI Gateway route this agent calls for generation — same route every other pattern
# in this repo uses, since this is a reference example rather than a production workload that
# needs its own provisioned model.
GATEWAY_ROUTE = "gpt-oss-120b"

# The MLflow AI Gateway route this agent calls for embeddings — a *second* gateway route,
# provisioned separately from GATEWAY_ROUTE (see
# packages/mlflow-server/scripts/provision_gateway_route.py's embeddings-route provisioning),
# since embedding and chat are different upstream models. Set SELFHOSTED_EMBEDDING_MODEL_NAME /
# EMBEDDING_GATEWAY_ROUTE_NAME in .env to this same value so the provisioned route name matches
# what this constant calls — same convention GATEWAY_ROUTE/GATEWAY_ROUTE_NAME already uses.
EMBEDDING_GATEWAY_ROUTE = "text-embedding"

# Must match packages/milvus/scripts/provision_collections.py's collection-naming rule
# (<seed-file-stem>.replace("-", "_")) for packages/milvus/collections/basic-rag-agent.jsonl —
# Milvus collection names can't contain hyphens.
COLLECTION_NAME = "basic_rag_agent"

# Retrieval k — see basic-rag.md's own note on the k tradeoff (too low misses context, too high
# floods the prompt with noise).
DEFAULT_K = 4

# Returned by `generate` instead of calling the model at all when retrieval comes back empty —
# see this module's docstring on basic-rag.md's "silent failure on empty retrieval" warning.
NO_CONTEXT_ANSWER = "I don't have relevant context to answer that question."

# Scorers run continuously against a sampled slice of live production traces — see
# agents_common.observability.register_production_monitors, provisioned via
# packages/mlflow-server/scripts/provision_monitors.py. `grounded_in_context`'s guideline text is
# loaded from packages/mlflow-server/judges/basic-rag-agent-grounded_in_context.txt, the same
# source that eval suite loads it from — single source of truth, see agents_common.judges.
PRODUCTION_SCORERS: list[tuple[Any, float]] = build_production_scorers(
    GATEWAY_ROUTE, [("grounded_in_context", "basic-rag-agent-grounded_in_context")]
)

# The alias provisioning points at the "live" version of the generation prompt — see
# packages/mlflow-server/scripts/provision_prompts.py, which registers this agent's single
# prompt from packages/mlflow-server/prompts/basic-rag-agent.txt. Only one prompt (the
# generation step) — retrieval has no prompt of its own, it's a vector search.
_PROMPT_ALIAS = PRODUCTION_ALIAS


class _Document(Protocol):
    page_content: str


class Retriever(Protocol):
    """The shape `build_rag_graph`'s `retriever` override needs to satisfy.

    Matches `langchain_core.retrievers.BaseRetriever`'s `.invoke()` signature structurally
    (Protocol, not a subclass requirement) — a real `langchain_milvus.Milvus(...).as_retriever()`
    satisfies this automatically; tests pass a lightweight fake instead. See `build_rag_graph`'s
    `retriever` parameter.
    """

    def invoke(self, query: str) -> list[_Document]:
        """Return the retrieved documents for `query`."""
        ...


class RagState(TypedDict):
    """State threaded through the graph.

    `retrieved_chunks` stores plain strings (`Document.page_content`), not `Document` objects —
    `retrieve` narrows immediately on the way in, so nothing downstream (including tests) needs
    to construct or import a `langchain_core.documents.Document`.
    """

    question: str
    retrieved_chunks: list[str]
    answer: str


def load_rag_prompt_version(*, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch this agent's generation prompt version from the MLflow prompt registry.

    Thin wrapper around `agents_common.prompts.load_prompt_version` binding this agent's own
    registry name and experiment. See packages/mlflow-server/scripts/provision_prompts.py, which
    registers `EXPERIMENT_NAME`'s prompt from packages/mlflow-server/prompts/basic-rag-agent.txt.

    Returns the full `PromptVersion` (not just its text) so a caller running the agent can pass
    it to `link_prompt_to_trace` afterwards — see `basic_rag_agent.__main__` for the intended
    usage.
    """
    return load_prompt_version(EXPERIMENT_NAME, experiment_name=EXPERIMENT_NAME, alias=alias)


def load_rag_prompt(*, alias: str = _PROMPT_ALIAS) -> str:
    """Fetch this agent's generation prompt text from the MLflow prompt registry.

    Thin wrapper around `load_rag_prompt_version` for callers that only need the text (e.g.
    `build_rag_graph`'s default path) and don't need to link the version to a trace afterwards.
    """
    return prompt_text(load_rag_prompt_version(alias=alias))


def link_prompt_to_trace(prompt_version: PromptVersion, trace_id: str | None) -> None:
    """Link the generation prompt version to a trace so the MLflow UI's trace view shows it.

    Thin wrapper around `agents_common.prompts.link_prompts_to_trace` for this agent's single
    prompt — see that function's docstring for `trace_id` semantics.
    """
    link_prompts_to_trace([prompt_version], trace_id)


def _build_default_retriever(embedding_gateway_route: str, milvus_uri: str, k: int) -> Retriever:
    vector_store = Milvus(
        embedding_function=get_embeddings(embedding_gateway_route),
        collection_name=COLLECTION_NAME,
        connection_args={"uri": milvus_uri},
    )
    return vector_store.as_retriever(search_kwargs={"k": k})  # type: ignore[return-value]


def build_rag_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    embedding_gateway_route: str = EMBEDDING_GATEWAY_ROUTE,
    milvus_uri: str | None = None,
    k: int = DEFAULT_K,
    rag_prompt: str | None = None,
    retriever: Retriever | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the basic RAG workflow.

    The caller owns the checkpointer's lifecycle, same convention as
    routing_agent.graph.build_router.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration)
            or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route the generation step calls. Defaults to this
            package's own `GATEWAY_ROUTE`; overridable for tests against a different route.
        embedding_gateway_route: MLflow AI Gateway route retrieval embeds queries with. Ignored
            if `retriever` is passed directly. Defaults to this package's own
            `EMBEDDING_GATEWAY_ROUTE`.
        milvus_uri: Overrides `Settings.milvus_uri`. Ignored if `retriever` is passed directly.
        k: Number of chunks to retrieve per query. Ignored if `retriever` is passed directly.
        rag_prompt: Overrides the registry-fetched generation prompt. Defaults to `None`, which
            fetches the current `production`-aliased prompt via `load_rag_prompt()`. Pass a
            literal string in tests that need a hermetic build with no MLflow prompt-registry
            dependency.
        retriever: Overrides the default Milvus-backed retriever entirely — anything satisfying
            `Retriever` (structurally, a `langchain_core.retrievers.BaseRetriever` included).
            Pass a fake in tests that need a hermetic build with no real Milvus/embeddings-route
            dependency; the real `langchain_milvus.Milvus` retriever is only constructed (and
            only needs `langchain-milvus`/`pymilvus` importable) when this is left `None`.

    Returns:
        A compiled LangGraph graph, invoked with `{"question": ..., "retrieved_chunks": [],
        "answer": ""}`.
    """
    model = get_chat_model(gateway_route)
    prompt = rag_prompt if rag_prompt is not None else load_rag_prompt()
    active_retriever = (
        retriever
        if retriever is not None
        else _build_default_retriever(
            embedding_gateway_route, milvus_uri or get_settings().milvus_uri, k
        )
    )

    def retrieve(state: RagState) -> dict[str, list[str]]:
        documents = active_retriever.invoke(state["question"])
        chunks = [doc.page_content for doc in documents]
        _logger.info("retrieved", chunk_count=len(chunks))
        return {"retrieved_chunks": chunks}

    def generate(state: RagState) -> dict[str, str]:
        if not state["retrieved_chunks"]:
            _logger.info("generated", grounded=False)
            return {"answer": NO_CONTEXT_ANSWER}

        context = "\n\n".join(state["retrieved_chunks"])
        response = model.invoke(f"{prompt}\n\nContext:\n{context}\n\nQuestion: {state['question']}")
        _logger.info("generated", grounded=True, chunk_count=len(state["retrieved_chunks"]))
        return {"answer": str(response.content)}

    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str) -> dict[str, Any]:
    """Build the `.invoke()` config for a thread.

    Every call site that runs the compiled graph should route through this, same convention as
    react_agent.graph.invoke_config / routing_agent.graph.invoke_config.
    """
    return {"configurable": {"thread_id": thread_id}}
