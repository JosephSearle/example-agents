"""Builds the Self-RAG workflow graph.

Per `docs/patterns/rag/self-rag.md`, this reuses Corrective RAG's document-grading loop
(`retrieve` -> `grade_documents` -> retry-or-generate) and adds a second, post-generation grading
step: the generated answer itself is graded for **groundedness** (is it actually supported by the
retrieved documents?) and **usefulness** (does it actually answer the question?) — two
independent signals, per the doc's own warning, requiring different corrective actions:
ungrounded -> regenerate from the same documents; grounded-but-not-useful -> re-retrieve with a
rewritten question. This is a genuine cycle (not just branching, contrast Corrective RAG), so it
needs two independent retry caps — `MAX_REGENERATE` for the generation-side loop, `MAX_RETRIES`
(shared naming/role with `corrective_rag_agent`) for the retrieval-side loop — each capped
separately so a persistent failure in one dimension can't be masked by budget shared with the
other. Per this module's own "infinite-loop risk" being the doc's headline caveat, both caps are
load-bearing, not decorative.

**Deviation from the doc** (inherited from `corrective_rag_agent`, see that module's docstring):
no external web search — the retrieval-side retry rewrites the question and retries against the
same Milvus collection.

**Advanced-section note**: the doc's `should_retrieve` conditional entry point (skip retrieval
entirely for some questions) is an intentional non-goal here — that's what `adaptive-rag-agent`'s
`no_retrieval` branch is for; duplicating it here would blur the two patterns' distinct teaching
points.

Reuses `basic_rag_agent`'s Milvus collection (`COLLECTION_NAME`) and embeddings route, and
duplicates (rather than imports) `corrective_rag_agent`'s `retrieve`/`grade_documents`/
`decide_to_generate`/`transform_query` node shapes — see that module's docstring for why node
functions are duplicated rather than shared across RAG patterns in this repo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

from agents_common import get_chat_model, get_embeddings, get_settings
from agents_common.judges import build_production_scorers
from agents_common.prompts import (
    PRODUCTION_ALIAS,
    link_prompts_to_trace as _link_prompts_to_trace,
    load_prompt_version,
    prompt_text,
)
from basic_rag_agent.graph import COLLECTION_NAME
from langchain_milvus import Milvus
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
    "EMBEDDING_GATEWAY_ROUTE",
    "GATEWAY_ROUTE",
    "MAX_REGENERATE",
    "MAX_RETRIES",
    "NO_CONTEXT_ANSWER",
    "PRODUCTION_SCORERS",
    "PROMPT_NAMES",
    "Retriever",
    "SelfRagState",
    "build_rag_graph",
    "invoke_config",
    "link_prompts_to_trace",
    "load_rag_prompt",
    "load_rag_prompt_version",
    "prompt_text",
]

EXPERIMENT_NAME = "self-rag-agent"
GATEWAY_ROUTE = "gpt-oss-120b"
EMBEDDING_GATEWAY_ROUTE = "text-embedding"

# Retrieval-side retry cap — same role as corrective_rag_agent.MAX_RETRIES.
MAX_RETRIES = 2

# Generation-side retry cap — regenerate-on-ungrounded, independent of the retrieval-side cap.
MAX_REGENERATE = 2

PROMPT_NAMES = (
    "grade_documents",
    "transform_query",
    "generate",
    "hallucination_grader",
    "answer_grader",
)

NO_CONTEXT_ANSWER = "I don't have relevant context to answer that question."

PRODUCTION_SCORERS: list[tuple[Any, float]] = build_production_scorers(
    GATEWAY_ROUTE, [("grounded_in_context", "self-rag-agent-grounded_in_context")]
)

_PROMPT_ALIAS = PRODUCTION_ALIAS


class _Document(Protocol):
    page_content: str


class Retriever(Protocol):
    """The shape `build_rag_graph`'s `retriever` override needs to satisfy.

    See `basic_rag_agent.graph.Retriever`, which this mirrors exactly.
    """

    def invoke(self, query: str) -> list[_Document]:
        """Return the retrieved documents for `query`."""
        ...


class DocumentGrade(BaseModel):
    """Structured output for the per-document relevance grader.

    Same shape as `corrective_rag_agent.graph.DocumentGrade`.
    """

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to answering the question, else 'no'."
    )


class HallucinationGrade(BaseModel):
    """Structured output for the groundedness grader.

    Is the answer actually supported by the retrieved documents?
    """

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer is grounded in / supported by the documents, else 'no'."
    )


class AnswerUsefulnessGrade(BaseModel):
    """Structured output for the usefulness grader.

    Does the answer actually address the question, independent of whether it's grounded?
    """

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer actually addresses the question, else 'no'."
    )


class SelfRagState(TypedDict):
    """State threaded through the graph.

    `retry_count` (retrieval-side) and `regenerate_count` (generation-side) are independent
    counters — see this module's docstring on why grounded-but-useless and ungrounded need
    separate budgets. `grounded`/`useful` are written by `grade_generation` and read by the
    `grade_generation_v_documents_and_question` conditional edge.
    """

    question: str
    original_question: str
    documents: list[str]
    documents_sufficient: bool
    retry_count: int
    answer: str
    grounded: bool
    useful: bool
    regenerate_count: int


def load_rag_prompt_version(step: str, *, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch one step's prompt version from the MLflow prompt registry.

    Args:
        step: One of `PROMPT_NAMES`.
        alias: Prompt registry alias to load. Defaults to the production alias.
    """
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


def _build_default_retriever(embedding_gateway_route: str, milvus_uri: str, k: int) -> Retriever:
    vector_store = Milvus(
        embedding_function=get_embeddings(embedding_gateway_route),
        collection_name=COLLECTION_NAME,
        connection_args={"uri": milvus_uri},
    )
    return vector_store.as_retriever(search_kwargs={"k": k})  # type: ignore[return-value]


def _build_retrieval_loop_nodes(
    *,
    model: Any,
    document_grader: Any,
    active_prompts: dict[str, str],
    active_retriever: Retriever,
    max_retries: int,
) -> tuple[Any, Any, Any, Any]:
    """Build the `retrieve`/`grade_documents`/`decide_to_generate`/`transform_query` callables.

    Pulled out of `build_rag_graph` to keep that function's own statement count reasonable; this
    quartet is otherwise identical to `corrective_rag_agent`'s (duplicated, not imported — see
    this module's docstring).
    """

    def retrieve(state: SelfRagState) -> dict[str, list[str]]:
        documents = active_retriever.invoke(state["question"])
        chunks = [doc.page_content for doc in documents]
        _logger.info("retrieved", chunk_count=len(chunks), retry_count=state["retry_count"])
        return {"documents": chunks}

    def grade_documents(state: SelfRagState) -> dict[str, list[str] | bool]:
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

    def decide_to_generate(state: SelfRagState) -> str:
        if state["documents_sufficient"]:
            return "generate"
        if state["retry_count"] < max_retries:
            return "transform_query"
        _logger.warning(
            "self_rag_retrieval_cap_reached",
            retry_count=state["retry_count"],
            max_retries=max_retries,
        )
        return "generate"

    def transform_query(state: SelfRagState) -> dict[str, str | int]:
        response = model.invoke(
            f"{active_prompts['transform_query']}\n\nOriginal question: {state['original_question']}\n"
            f"Previous rewrite: {state['question']}"
        )
        rewritten = str(response.content)
        _logger.info("transformed_query", retry_count=state["retry_count"] + 1)
        return {"question": rewritten, "retry_count": state["retry_count"] + 1}

    return retrieve, grade_documents, decide_to_generate, transform_query


def _build_generation_loop_nodes(
    *,
    model: Any,
    hallucination_grader: Any,
    answer_grader: Any,
    active_prompts: dict[str, str],
    max_retries: int,
    max_regenerate: int,
) -> tuple[Any, Any, Any]:
    """Build the `generate`/`grade_generation`-plus-routing callables.

    Pulled out of `build_rag_graph` to keep that function's own statement count reasonable.
    """

    def generate(state: SelfRagState) -> dict[str, str]:
        if not state["documents"]:
            _logger.info("generated", grounded=False)
            return {"answer": NO_CONTEXT_ANSWER}

        context = "\n\n".join(state["documents"])
        response = model.invoke(
            f"{active_prompts['generate']}\n\nContext:\n{context}\n\n"
            f"Question: {state['original_question']}"
        )
        _logger.info("generated", chunk_count=len(state["documents"]))
        return {"answer": str(response.content)}

    def grade_generation(state: SelfRagState) -> dict[str, bool | int]:
        context = "\n\n".join(state["documents"])
        hallucination = hallucination_grader.invoke(
            f"{active_prompts['hallucination_grader']}\n\nDocuments:\n{context}\n\n"
            f"Answer: {state['answer']}"
        )
        grounded = hallucination.binary_score == "yes"

        useful = False
        if grounded:
            usefulness = answer_grader.invoke(
                f"{active_prompts['answer_grader']}\n\nQuestion: {state['original_question']}\n\n"
                f"Answer: {state['answer']}"
            )
            useful = usefulness.binary_score == "yes"

        updates: dict[str, bool | int] = {"grounded": grounded, "useful": useful}
        if not grounded:
            # The regenerate-triggering failure is detected here, so the count is incremented at
            # detection time rather than in a separate "corrective action" node (contrast
            # `transform_query`, which increments `retry_count` post-decision) — there's no
            # distinct node for "regenerate," it's the same `generate` node looped back into.
            updates["regenerate_count"] = state["regenerate_count"] + 1
        _logger.info("graded_generation", grounded=grounded, useful=useful)
        return updates

    def grade_generation_v_documents_and_question(state: SelfRagState) -> str:
        if not state["grounded"]:
            # regenerate_count was already incremented in grade_generation (detection-time, not
            # decision-time — see that node's comment), so this is a post-increment check: `<=`
            # here permits exactly `max_regenerate` regenerate attempts, matching
            # decide_to_generate's pre-increment `<` check permitting `max_retries` retries there.
            if state["regenerate_count"] <= max_regenerate:
                return "generate"
            _logger.warning(
                "self_rag_regenerate_cap_reached",
                regenerate_count=state["regenerate_count"],
                max_regenerate=max_regenerate,
            )
            return END
        if not state["useful"]:
            if state["retry_count"] < max_retries:
                return "transform_query"
            _logger.warning(
                "self_rag_useful_retry_cap_reached",
                retry_count=state["retry_count"],
                max_retries=max_retries,
            )
            return END
        return END

    return generate, grade_generation, grade_generation_v_documents_and_question


def build_rag_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    embedding_gateway_route: str = EMBEDDING_GATEWAY_ROUTE,
    milvus_uri: str | None = None,
    k: int = 4,
    max_retries: int = MAX_RETRIES,
    max_regenerate: int = MAX_REGENERATE,
    prompts: dict[str, str] | None = None,
    retriever: Retriever | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the Self-RAG workflow.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration)
            or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route called for every grading/rewriting/generation step.
        embedding_gateway_route: MLflow AI Gateway route retrieval embeds queries with. Ignored
            if `retriever` is passed directly.
        milvus_uri: Overrides `Settings.milvus_uri`. Ignored if `retriever` is passed directly.
        k: Number of chunks to retrieve per attempt.
        max_retries: Overrides `MAX_RETRIES` — the retrieval-side retry cap.
        max_regenerate: Overrides `MAX_REGENERATE` — the generation-side retry cap.
        prompts: Overrides the registry-fetched prompts, keyed by `PROMPT_NAMES`. Defaults to
            `None`, which fetches each step's current `production`-aliased prompt. Pass literal
            strings in tests that need a hermetic build with no MLflow prompt-registry dependency.
        retriever: Overrides the default Milvus-backed retriever. Pass a fake in tests.

    Returns:
        A compiled LangGraph graph, invoked with `{"question": ..., "original_question": ...,
        "documents": [], "documents_sufficient": False, "retry_count": 0, "answer": "",
        "grounded": False, "useful": False, "regenerate_count": 0}`.
    """
    model = get_chat_model(gateway_route)
    document_grader = model.with_structured_output(DocumentGrade)
    hallucination_grader = model.with_structured_output(HallucinationGrade)
    answer_grader = model.with_structured_output(AnswerUsefulnessGrade)
    active_prompts = (
        prompts if prompts is not None else {step: load_rag_prompt(step) for step in PROMPT_NAMES}
    )
    active_retriever = (
        retriever
        if retriever is not None
        else _build_default_retriever(
            embedding_gateway_route, milvus_uri or get_settings().milvus_uri, k
        )
    )

    retrieve, grade_documents, decide_to_generate, transform_query = _build_retrieval_loop_nodes(
        model=model,
        document_grader=document_grader,
        active_prompts=active_prompts,
        active_retriever=active_retriever,
        max_retries=max_retries,
    )
    generate, grade_generation, grade_generation_v_documents_and_question = (
        _build_generation_loop_nodes(
            model=model,
            hallucination_grader=hallucination_grader,
            answer_grader=answer_grader,
            active_prompts=active_prompts,
            max_retries=max_retries,
            max_regenerate=max_regenerate,
        )
    )

    graph = StateGraph(SelfRagState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("transform_query", transform_query)
    graph.add_node("generate", generate)
    graph.add_node("grade_generation", grade_generation)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("generate", "grade_generation")
    graph.add_conditional_edges(
        "grade_generation",
        grade_generation_v_documents_and_question,
        {"generate": "generate", "transform_query": "transform_query", END: END},
    )

    return graph.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str) -> dict[str, Any]:
    """Build the `.invoke()` config for a thread."""
    return {"configurable": {"thread_id": thread_id}}
