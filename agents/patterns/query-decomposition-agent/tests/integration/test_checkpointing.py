"""Integration test: the graph's state actually survives a rebuild via Postgres, with retrieval
(once per sub-question) against the real, seeded Milvus collection (decompose/generate/synthesize
faked — see conftest.py).

Run with `pytest -m integration` against live Postgres AND Milvus (`docker compose up -d postgres
milvus-standalone` + `make provision-milvus-collections`)."""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

import pytest
from query_decomposition_agent.graph import build_rag_graph, invoke_config

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

pytestmark = pytest.mark.integration


def test_thread_state_persists_across_graph_rebuilds(
    postgres_checkpointer: PostgresSaver,
    real_milvus_collection: None,
    fake_chat_model: object,
    prompts: dict[str, str],
) -> None:
    _ = real_milvus_collection  # fixture only exists to skip early if Milvus isn't seeded
    thread_id = str(uuid.uuid4())
    config = invoke_config(thread_id)
    initial_state = {
        "question": "What framework tiers does this repo use?",
        "sub_questions": [],
        "sub_answers": [],
        "answer": "",
    }

    first_graph = build_rag_graph(checkpointer=postgres_checkpointer, prompts=prompts)
    first_graph.invoke(initial_state, config=config)

    # Simulate a process restart: a brand-new compiled graph, same checkpointer/thread.
    second_graph = build_rag_graph(checkpointer=postgres_checkpointer, prompts=prompts)
    state = second_graph.get_state(config)

    assert state.values["sub_questions"] == ["first sub-question", "second sub-question"]
    assert len(state.values["sub_answers"]) == 2
    assert all(answer for answer in state.values["sub_answers"]), (
        "expected real Milvus retrieval to return chunks for each fixed sub-question"
    )
    assert state.values["answer"] == "a canned grounded answer"
