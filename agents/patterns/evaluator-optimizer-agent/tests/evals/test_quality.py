"""MLflow GenAI eval suite for the evaluator-optimizer workflow.

Calls a real model, so this is gated out of the default CI run (see `.github/workflows/eval.yml`
and the `eval` marker) — run explicitly with `pytest -m eval`. Results land in MLflow as a run
under this agent's own `evaluator_optimizer_agent.EXPERIMENT_NAME` experiment.

The dataset lives in MLflow's dataset registry, seeded from
`packages/mlflow-server/datasets/evaluator-optimizer-agent.jsonl` via `make provision-datasets`.
Each record's `expectations.expected_approved` is whether the loop should have reached its own
evaluator's approval within `DEFAULT_MAX_ITERATIONS` — scored directly against the graph's own
`approved` output (exact match) rather than by a second LLM judge, same convention as
routing-agent's `correct_category` scorer.
"""

from __future__ import annotations

import os

from agents_common import configure_mlflow
from agents_common.config import get_settings
from agents_common.judges import load_judge_guidelines
from evaluator_optimizer_agent.graph import (
    EXPERIMENT_NAME,
    GATEWAY_ROUTE,
    build_evaluator_optimizer_graph,
    invoke_config,
)
from langgraph.checkpoint.memory import InMemorySaver
import mlflow
from mlflow.genai.datasets import get_dataset
from mlflow.genai.scorers import Guidelines, scorer
import pytest

pytestmark = pytest.mark.eval

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(task: str, criteria: str) -> dict[str, object]:
    checkpointer = InMemorySaver()
    graph = build_evaluator_optimizer_graph(checkpointer=checkpointer)
    config = invoke_config(f"{task}::{criteria}")
    result = graph.invoke(
        {
            "task": task,
            "criteria": criteria,
            "response": "",
            "feedback": "",
            "approved": False,
            "iteration": 0,
        },
        config=config,
    )
    return {"approved": result["approved"], "response": result["response"]}


@scorer
def reached_approval(outputs: dict[str, object], expectations: dict[str, bool]) -> bool:
    """Did the loop's own evaluator approve the response within the iteration cap?"""
    return outputs["approved"] == expectations["expected_approved"]


def test_evaluator_optimizer_agent_eval_suite() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name="evaluator-optimizer-agent-eval"):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=_predict_fn,
            scorers=[
                reached_approval,
                Guidelines(
                    name="meets_criteria",
                    guidelines=load_judge_guidelines("evaluator-optimizer-agent-meets_criteria"),
                    model=_JUDGE_MODEL_URI,
                ),
            ],
        )

    # Threshold, not a hardcoded 100% — LLM-judged scores are noisy by nature, and the loop's own
    # evaluator doesn't always converge within the iteration cap. Tune once real eval history
    # exists in MLflow (see the MLflow UI's evaluation-runs tab for this experiment).
    assert results.metrics["reached_approval/mean"] >= 0.6
    assert results.metrics["meets_criteria/mean"] >= 0.7
