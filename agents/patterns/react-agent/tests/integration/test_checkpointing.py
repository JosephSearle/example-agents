"""Integration test: the agent's conversation state actually survives a rebuild via Postgres.

This is the point of the checkpointing pattern — run with `pytest -m integration` against a
live Postgres (`docker compose up -d postgres`). Uses `fake_chat_model` (see conftest.py) rather
than a real gateway model: this test is about Postgres round-tripping checkpoint state, not
model behavior, so it shouldn't need network access to a live model to run. Passes an explicit
`system_prompt` to `build_agent` for the same reason: `build_agent`'s default path fetches the
prompt from MLflow's prompt registry (see `react_agent.graph.load_system_prompt`), which would
otherwise pull in a live-MLflow dependency this test — and CI's `integration.yml`, which
deliberately provisions only Postgres — isn't meant to need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

import pytest
from react_agent.graph import build_agent, invoke_config

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

pytestmark = pytest.mark.integration

_TEST_SYSTEM_PROMPT = "You are a helpful assistant."


def test_thread_state_persists_across_agent_rebuilds(
    postgres_checkpointer: PostgresSaver,
    fake_chat_model: object,
) -> None:
    _ = fake_chat_model  # fixture patches get_chat_model as a side effect; unused directly
    thread_id = str(uuid.uuid4())
    config = invoke_config(thread_id)

    first_agent = build_agent(checkpointer=postgres_checkpointer, system_prompt=_TEST_SYSTEM_PROMPT)
    first_agent.invoke(
        {"messages": [{"role": "user", "content": "Remember the number 8675309."}]}, config=config
    )  # type: ignore[arg-type]

    # Simulate a process restart: a brand-new compiled graph, same checkpointer/thread.
    second_agent = build_agent(
        checkpointer=postgres_checkpointer, system_prompt=_TEST_SYSTEM_PROMPT
    )
    state = second_agent.get_state(config)  # type: ignore[arg-type]

    assert state.values["messages"], (
        "expected the first turn's messages to be recovered from Postgres"
    )
