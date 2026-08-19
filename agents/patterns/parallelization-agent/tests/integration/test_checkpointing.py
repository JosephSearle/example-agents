"""Integration test: the graph's state actually survives a rebuild via Postgres.

Run with `pytest -m integration` against a live Postgres (`docker compose up -d postgres`). Uses
`fake_chat_model` (see conftest.py) rather than a real gateway model: this test is about Postgres
round-tripping checkpoint state, not model behavior. Passes explicit `section_prompts` to
`build_sectioning_graph` for the same reason: the default path fetches prompts from MLflow's
prompt registry, which would otherwise pull in a live-MLflow dependency this test doesn't need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from parallelization_agent.graph import build_sectioning_graph, invoke_config
import pytest

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

pytestmark = pytest.mark.integration


def test_thread_state_persists_across_graph_rebuilds(
    postgres_checkpointer: PostgresSaver,
    fake_chat_model: object,
    section_prompts: dict[str, str],
) -> None:
    _ = fake_chat_model  # fixture patches get_chat_model as a side effect; unused directly
    thread_id = str(uuid.uuid4())
    config = invoke_config(thread_id)

    first_graph = build_sectioning_graph(
        checkpointer=postgres_checkpointer, section_prompts=section_prompts
    )
    first_graph.invoke(
        {
            "incident_text": "the database ran out of disk space",
            "summary": "",
            "severity": "",
            "action_items": [],
            "report": "",
        },
        config=config,
    )

    # Simulate a process restart: a brand-new compiled graph, same checkpointer/thread.
    second_graph = build_sectioning_graph(
        checkpointer=postgres_checkpointer, section_prompts=section_prompts
    )
    state = second_graph.get_state(config)

    assert state.values["severity"] == "critical"
    assert state.values["report"], "expected the aggregated report to be recovered from Postgres"
