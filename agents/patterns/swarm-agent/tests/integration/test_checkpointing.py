"""Integration test: the swarm's active-agent state actually survives a rebuild via Postgres.

Run with `pytest -m integration` against a live Postgres (`docker compose up -d postgres`). Uses
`fake_chat_model` (see conftest.py) rather than a real gateway model: this test is about Postgres
round-tripping the swarm's persistent-active-agent state across two turns — a follow-up message
in the same thread resuming with whichever specialist last took control, per
docs/patterns/agent/swarm-handoffs.md's own worked example — not model behavior. Passes explicit
`agent_prompts` to `build_swarm` for the same reason: the default path fetches prompts from
MLflow's prompt registry, which would otherwise pull in a live-MLflow dependency this test
doesn't need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

import pytest
from swarm_agent.graph import build_swarm, invoke_config

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

pytestmark = pytest.mark.integration


def test_thread_resumes_with_the_last_active_agent_across_rebuilds(
    postgres_checkpointer: PostgresSaver,
    fake_chat_model: object,
    agent_prompts: dict[str, str],
) -> None:
    _ = fake_chat_model  # fixture patches get_chat_model as a side effect; unused directly
    thread_id = str(uuid.uuid4())
    config = invoke_config(thread_id)

    first_swarm = build_swarm(checkpointer=postgres_checkpointer, agent_prompts=agent_prompts)
    first_result = first_swarm.invoke(  # type: ignore[arg-type]
        {"messages": [{"role": "user", "content": "I'd like a refund for INV-1002."}]},
        config=config,
    )

    assert first_result["messages"][-1].content == "Refunded $19.99 for INV-1002."

    # Simulate a process restart: a brand-new compiled graph, same checkpointer/thread. If the
    # active-agent state hadn't survived in Postgres, this second turn would restart at
    # DEFAULT_ACTIVE_AGENT ("triage") instead of resuming directly with "billing" — the fake
    # model's canned sequence (conftest.py) only makes sense if it resumes with billing.
    second_swarm = build_swarm(checkpointer=postgres_checkpointer, agent_prompts=agent_prompts)
    second_result = second_swarm.invoke(  # type: ignore[arg-type]
        {"messages": [{"role": "user", "content": "What about INV-1001?"}]}, config=config
    )

    assert second_result["messages"][-1].content == "INV-1001: $49.99"
