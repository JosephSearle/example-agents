"""Shared Milvus retriever construction for the RAG pattern family.

Every RAG-family pattern (basic/adaptive/corrective/self/retrieve-rerank/query-decomposition-rag)
retrieves from its own Milvus collection via the same `Milvus(embedding_function=get_embeddings(...),
...).as_retriever()` wiring, differing only in which collection and embedding route they name.
Originally re-implemented once per agent; pulled up here so a future change to retriever
construction (search_type, connection pooling, retry/backoff) is a one-file fix instead of a
six-file one.
"""

from __future__ import annotations

from typing import Protocol

from langchain_milvus import Milvus

from agents_common.models import get_embeddings

# Returned by a RAG pattern's `generate` node instead of calling the model at all when retrieval
# comes back empty, rather than forwarding an empty context block — an empty context risks a
# confident hallucination instead of an honest "I don't know" (see docs/patterns/rag/basic-rag.md's
# "silent failure on empty retrieval" warning). Shared across the RAG family so the fallback
# wording has one source of truth instead of five independently-typed copies.
NO_CONTEXT_ANSWER = "I don't have relevant context to answer that question."


class _Document(Protocol):
    page_content: str


class Retriever(Protocol):
    """The shape a RAG pattern's `retriever` override needs to satisfy.

    Matches `langchain_core.retrievers.BaseRetriever`'s `.invoke()` signature structurally
    (Protocol, not a subclass requirement) — a real `langchain_milvus.Milvus(...).as_retriever()`
    satisfies this automatically; tests pass a lightweight fake instead.
    """

    def invoke(self, query: str) -> list[_Document]:
        """Return the retrieved documents for `query`."""
        ...


def build_milvus_retriever(
    *, collection_name: str, embedding_gateway_route: str, milvus_uri: str, k: int
) -> Retriever:
    """Build the default Milvus-backed retriever for a RAG pattern's `k` nearest chunks.

    Args:
        collection_name: The Milvus collection to query — must match
            `packages/milvus/scripts/provision_collections.py`'s collection-naming rule.
        embedding_gateway_route: MLflow AI Gateway route used to embed the query.
        milvus_uri: Milvus connection URI.
        k: Number of chunks to retrieve per query.
    """
    vector_store = Milvus(
        embedding_function=get_embeddings(embedding_gateway_route),
        collection_name=collection_name,
        connection_args={"uri": milvus_uri},
    )
    return vector_store.as_retriever(search_kwargs={"k": k})  # type: ignore[return-value]
