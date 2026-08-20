"""Integration test: the graph's state actually survives a rebuild via Postgres.

Run with `pytest -m integration` against a live Postgres (`docker compose up -d postgres`). Uses
`fake_chat_model` (see conftest.py) rather than a real gateway model: this test is about Postgres
round-tripping checkpoint state, not model behavior. Passes explicit `agent_prompts` to
`build_mesh_graph` for the same reason: the default path fetches prompts from MLflow's prompt
registry, which would otherwise pull in a live-MLflow dependency this test doesn't need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from network_mesh_agent.graph import build_mesh_graph, invoke_config
import pytest

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

pytestmark = pytest.mark.integration


def test_thread_state_persists_across_graph_rebuilds(
    postgres_checkpointer: PostgresSaver,
    fake_chat_model: object,
    agent_prompts: dict[str, str],
) -> None:
    _ = fake_chat_model  # fixture patches get_chat_model as a side effect; unused directly
    thread_id = str(uuid.uuid4())
    config = invoke_config(thread_id)
    initial_state = {
        "task": "some task",
        "messages": [],
        "needs_critique": False,
        "needs_more_research": False,
        "research_rounds": 0,
        "final_answer": "",
    }

    first_graph = build_mesh_graph(checkpointer=postgres_checkpointer, agent_prompts=agent_prompts)
    first_graph.invoke(initial_state, config=config)

    # Simulate a process restart: a brand-new compiled graph, same checkpointer/thread.
    second_graph = build_mesh_graph(checkpointer=postgres_checkpointer, agent_prompts=agent_prompts)
    state = second_graph.get_state(config)

    assert state.values["final_answer"] == "a canned final answer"
    assert state.values["research_rounds"] == 1
    assert any(m["role"] == "researcher" for m in state.values["messages"])
