"""Integration test: the supervisor's conversation state actually survives a rebuild via Postgres.

Run with `pytest -m integration` against a live Postgres (`docker compose up -d postgres`). Uses
`fake_chat_model` (see conftest.py) rather than a real gateway model: this test is about Postgres
round-tripping checkpoint state — and exercising a full supervisor → sub-agent → tool → supervisor
round trip against canned model responses — not model behavior. Passes explicit `agent_prompts`
to `build_supervisor` for the same reason: the default path fetches prompts from MLflow's prompt
registry, which would otherwise pull in a live-MLflow dependency this test doesn't need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

import pytest
from supervisor_agent.graph import build_supervisor, invoke_config

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

pytestmark = pytest.mark.integration


def test_thread_state_persists_across_supervisor_rebuilds(
    postgres_checkpointer: PostgresSaver,
    fake_chat_model: object,
    agent_prompts: dict[str, str],
) -> None:
    _ = fake_chat_model  # fixture patches get_chat_model as a side effect; unused directly
    thread_id = str(uuid.uuid4())
    config = invoke_config(thread_id)

    first_supervisor = build_supervisor(
        checkpointer=postgres_checkpointer, agent_prompts=agent_prompts
    )
    result = first_supervisor.invoke(  # type: ignore[arg-type]
        {"messages": [{"role": "user", "content": "What is 2+2?"}]}, config=config
    )

    assert result["messages"][-1].content == "The answer is 4."

    # Simulate a process restart: a brand-new compiled graph, same checkpointer/thread.
    second_supervisor = build_supervisor(
        checkpointer=postgres_checkpointer, agent_prompts=agent_prompts
    )
    state = second_supervisor.get_state(config)  # type: ignore[arg-type]

    assert state.values["messages"], (
        "expected the first turn's messages to be recovered from Postgres"
    )
