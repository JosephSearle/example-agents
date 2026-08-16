"""Integration test: the chain's state actually survives a rebuild via Postgres.

Run with `pytest -m integration` against a live Postgres (`docker compose up -d postgres`). Uses
`fake_chat_model` (see conftest.py) rather than a real gateway model: this test is about
Postgres round-tripping checkpoint state, not model behavior. Passes explicit `step_prompts` to
`build_chain` for the same reason: `build_chain`'s default path fetches prompts from MLflow's
prompt registry, which would otherwise pull in a live-MLflow dependency this test doesn't need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from prompt_chaining_agent.graph import build_chain, invoke_config
import pytest

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

pytestmark = pytest.mark.integration


def test_thread_state_persists_across_chain_rebuilds(
    postgres_checkpointer: PostgresSaver,
    fake_chat_model: object,
    step_prompts: dict[str, str],
) -> None:
    _ = fake_chat_model  # fixture patches get_chat_model as a side effect; unused directly
    thread_id = str(uuid.uuid4())
    config = invoke_config(thread_id)

    first_chain = build_chain(checkpointer=postgres_checkpointer, step_prompts=step_prompts)
    first_chain.invoke(
        {"topic": "some topic", "outline": "", "draft": "", "final": ""}, config=config
    )

    # Simulate a process restart: a brand-new compiled graph, same checkpointer/thread.
    second_chain = build_chain(checkpointer=postgres_checkpointer, step_prompts=step_prompts)
    state = second_chain.get_state(config)

    assert state.values["final"], "expected the chain's final output to be recovered from Postgres"
