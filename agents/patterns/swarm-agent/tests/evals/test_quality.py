"""MLflow GenAI eval suite for the swarm/handoffs workflow.

Calls a real model, so this is gated out of the default CI run (see `.github/workflows/eval.yml`
and the `eval` marker) — run explicitly with `pytest -m eval`. Results land in MLflow as a run
under this agent's own `swarm_agent.EXPERIMENT_NAME` experiment.

The dataset lives in MLflow's dataset registry, seeded from
`packages/mlflow-server/datasets/swarm-agent.jsonl` via `make provision-datasets`. Each record's
`expectations.expected_active_agent` is which agent should own the final response for that
message — scored directly (exact match against which agent's `name` produced the last message)
rather than by an LLM judge, same convention as routing-agent's `correct_category` scorer.
"""

from __future__ import annotations

import os

from agents_common import configure_mlflow
from agents_common.config import get_settings
from agents_common.judges import Safety, load_judge_guidelines, regression_subset
from langgraph.checkpoint.memory import InMemorySaver
import mlflow
from mlflow.genai.datasets import get_dataset
from mlflow.genai.scorers import Guidelines, scorer
import pytest
from swarm_agent.graph import EXPERIMENT_NAME, GATEWAY_ROUTE, build_swarm, invoke_config

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(message: str) -> dict[str, object]:
    checkpointer = InMemorySaver()
    swarm = build_swarm(checkpointer=checkpointer)
    config = invoke_config(message)
    result = swarm.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    last_message = result["messages"][-1]
    # `create_agent(..., name=...)` tags every message it produces with that agent's name (see
    # docs/patterns/agent/swarm-handoffs.md's worked example) — this is how we know which
    # specialist actually owned the final response, without inspecting handoff tool calls.
    active_agent = getattr(last_message, "name", None) or "triage"
    return {"active_agent": active_agent, "answer": str(last_message.content)}


@scorer
def correct_active_agent(outputs: dict[str, object], expectations: dict[str, str]) -> bool:
    """Did the expected specialist end up owning the final response?"""
    return outputs["active_agent"] == expectations["expected_active_agent"]


@pytest.mark.eval
def test_swarm_agent_eval_suite() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name="swarm-agent-eval"):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=_predict_fn,
            scorers=[
                correct_active_agent,
                Guidelines(
                    name="owns_handoff_conversation",
                    guidelines=load_judge_guidelines("swarm-agent-owns_handoff_conversation"),
                    model=_JUDGE_MODEL_URI,
                ),
                Safety(model=_JUDGE_MODEL_URI),  # type: ignore[no-untyped-call]
            ],
        )

    # Threshold, not a hardcoded 100% — LLM-judged scores are noisy by nature, and even the
    # exact-match active-agent scorer is allowed some slack for classifier variance. Tune once
    # real eval history exists in MLflow (see the MLflow UI's evaluation-runs tab for this
    # experiment).
    assert results.metrics["correct_active_agent/mean"] >= 0.7
    assert results.metrics["owns_handoff_conversation/mean"] >= 0.7


@pytest.mark.regression
def test_swarm_agent_regression() -> None:
    """Small, previously-verified-good subset (`tags: ["regression"]`); code-based-grader-first
    and thresholded near 100%, unlike the noisier LLM-judge-heavy capability suite above. Required
    on every PR — see docs/decisions/0002-eval-taxonomy.md.
    """
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)
    records = regression_subset(dataset)

    with mlflow.start_run(run_name="swarm-agent-regression"):
        results = mlflow.genai.evaluate(
            data=records,
            predict_fn=_predict_fn,
            scorers=[correct_active_agent],
        )

    assert results.metrics["correct_active_agent/mean"] >= 0.95
