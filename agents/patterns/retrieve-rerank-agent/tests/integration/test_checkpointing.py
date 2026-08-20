"""Integration test: the graph's state actually survives a rebuild via Postgres, with retrieval
and reranking against the real, seeded Milvus collection (reranker itself faked — see conftest.py).

Run with `pytest -m integration` against live Postgres AND Milvus (`docker compose up -d postgres
milvus-standalone` + `make provision-milvus-collections`)."""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

import pytest
from retrieve_rerank_agent.graph import build_rag_graph, invoke_config

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

pytestmark = pytest.mark.integration


def test_thread_state_persists_across_graph_rebuilds(
    postgres_checkpointer: PostgresSaver,
    real_milvus_collection: None,
    fake_chat_model: object,
    fake_reranker: object,
    rag_prompt: str,
) -> None:
    _ = real_milvus_collection  # fixture only exists to skip early if Milvus isn't seeded
    _ = fake_chat_model  # fixture patches get_chat_model as a side effect; unused directly
    thread_id = str(uuid.uuid4())
    config = invoke_config(thread_id)
    initial_state = {
        "question": "What framework tier does react-agent use?",
        "candidate_chunks": [],
        "reranked_chunks": [],
        "answer": "",
    }

    first_graph = build_rag_graph(
        checkpointer=postgres_checkpointer, rag_prompt=rag_prompt, reranker=fake_reranker
    )
    first_graph.invoke(initial_state, config=config)

    # Simulate a process restart: a brand-new compiled graph, same checkpointer/thread.
    second_graph = build_rag_graph(
        checkpointer=postgres_checkpointer, rag_prompt=rag_prompt, reranker=fake_reranker
    )
    state = second_graph.get_state(config)

    assert state.values["candidate_chunks"], "expected real Milvus retrieval to return chunks"
    assert state.values["reranked_chunks"], "expected the reranker step to produce chunks"
    assert state.values["answer"] == "a canned grounded answer"
