"""Integration test: the graph's state actually survives a rebuild via Postgres.

Run with `pytest -m integration` against a live Postgres (`docker compose up -d postgres`). Uses
`fake_chat_model` (see conftest.py) rather than a real gateway model: this test is about Postgres
round-tripping checkpoint state — in particular the loop-carried `iteration`/`feedback` fields —
not model behavior. Passes explicit `step_prompts` to `build_evaluator_optimizer_graph` for the
same reason: the default path fetches prompts from MLflow's prompt registry, which would
otherwise pull in a live-MLflow dependency this test doesn't need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import uuid

from evaluator_optimizer_agent.graph import build_evaluator_optimizer_graph, invoke_config
import pytest

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres import PostgresSaver

pytestmark = pytest.mark.integration


def test_thread_state_persists_across_graph_rebuilds(
    postgres_checkpointer: PostgresSaver,
    fake_chat_model: object,
    step_prompts: dict[str, str],
) -> None:
    _ = fake_chat_model  # fixture patches get_chat_model as a side effect; unused directly
    thread_id = str(uuid.uuid4())
    config = invoke_config(thread_id)

    first_graph = build_evaluator_optimizer_graph(
        checkpointer=postgres_checkpointer, step_prompts=step_prompts
    )
    first_graph.invoke(
        {
            "task": "some task",
            "criteria": "some criteria",
            "response": "",
            "feedback": "",
            "approved": False,
            "iteration": 0,
        },
        config=config,
    )

    # Simulate a process restart: a brand-new compiled graph, same checkpointer/thread.
    second_graph = build_evaluator_optimizer_graph(
        checkpointer=postgres_checkpointer, step_prompts=step_prompts
    )
    state = second_graph.get_state(config)

    # _FakeEvaluator (conftest.py) rejects the first attempt then approves the second, so the
    # loop should have run exactly two generate/evaluate cycles before exiting.
    assert state.values["iteration"] == 2
    assert state.values["approved"] is True
