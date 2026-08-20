"""Integration test: the graph's state actually survives a rebuild via Postgres, with retrieval
against the real, seeded Milvus collection.

Run with `pytest -m integration` against live Postgres AND Milvus (`docker compose up -d postgres
milvus-standalone` + `make provision-milvus-collections`). Uses `fake_chat_model` (see
conftest.py) rather than a real gateway chat model: this test is about Postgres round-tripping
checkpoint state and real retrieval working end-to-end, not chat model behavior. This is the one
pattern in this repo whose integration tests need a second live service beyond Postgres — see
this package's README for why (basic-rag.md's whole point is retrieval against a real, populated
collection).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from basic_rag_agent.graph import build_rag_graph, invoke_config
import pytest

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

pytestmark = pytest.mark.integration


def test_thread_state_persists_across_graph_rebuilds(
    postgres_checkpointer: PostgresSaver,
    real_milvus_collection: None,
    fake_chat_model: object,
    rag_prompt: str,
) -> None:
    _ = real_milvus_collection  # fixture only exists to skip early if Milvus isn't seeded
    _ = fake_chat_model  # fixture patches get_chat_model as a side effect; unused directly
    thread_id = str(uuid.uuid4())
    config = invoke_config(thread_id)
    initial_state = {
        "question": "What framework tier does react-agent use?",
        "retrieved_chunks": [],
        "answer": "",
    }

    first_graph = build_rag_graph(checkpointer=postgres_checkpointer, rag_prompt=rag_prompt)
    first_graph.invoke(initial_state, config=config)

    # Simulate a process restart: a brand-new compiled graph, same checkpointer/thread.
    second_graph = build_rag_graph(checkpointer=postgres_checkpointer, rag_prompt=rag_prompt)
    state = second_graph.get_state(config)

    assert state.values["retrieved_chunks"], "expected real Milvus retrieval to return chunks"
    assert state.values["answer"] == "a canned grounded answer"
