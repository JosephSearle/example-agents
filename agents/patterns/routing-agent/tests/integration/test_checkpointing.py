"""Integration test: the graph's state actually survives a rebuild via Postgres.

Run with `pytest -m integration` against a live Postgres (`docker compose up -d postgres`). Uses
`fake_chat_model` (see conftest.py) rather than a real gateway model: this test is about
Postgres round-tripping checkpoint state, not model behavior. Passes explicit `route_prompts` to
`build_router` for the same reason: `build_router`'s default path fetches prompts from MLflow's
prompt registry, which would otherwise pull in a live-MLflow dependency this test doesn't need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

import pytest
from routing_agent.graph import build_router, invoke_config

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

pytestmark = pytest.mark.integration


def test_thread_state_persists_across_router_rebuilds(
    postgres_checkpointer: PostgresSaver,
    fake_chat_model: object,
    route_prompts: dict[str, str],
) -> None:
    _ = fake_chat_model  # fixture patches get_chat_model as a side effect; unused directly
    thread_id = str(uuid.uuid4())
    config = invoke_config(thread_id)

    first_router = build_router(checkpointer=postgres_checkpointer, route_prompts=route_prompts)
    first_router.invoke(
        {"message": "the app crashes on startup", "category": "", "response": ""}, config=config
    )

    # Simulate a process restart: a brand-new compiled graph, same checkpointer/thread.
    second_router = build_router(checkpointer=postgres_checkpointer, route_prompts=route_prompts)
    state = second_router.get_state(config)

    assert state.values["category"] == "technical"
    assert state.values["response"], "expected the router's response to be recovered from Postgres"
