"""MLflow GenAI eval suite for the orchestrator-workers workflow.

Calls a real model, so this is gated out of the default CI run (see `.github/workflows/eval.yml`
and the `eval` marker) — run explicitly with `pytest -m eval`. Results land in MLflow as a run
under this agent's own `orchestrator_workers_agent.EXPERIMENT_NAME` experiment.

The dataset lives in MLflow's dataset registry, seeded from
`packages/mlflow-server/datasets/orchestrator-workers-agent.jsonl` via `make provision-datasets`.
Each record's `expectations.min_subtasks` is a lower bound, not an exact count — unlike
routing-agent's `correct_category` or map-reduce-agent's `correct_joke_count`, this pattern's
subtask count is genuinely decided by the orchestrator LLM per input, so there's no single
ground-truth count to match exactly against.
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
from orchestrator_workers_agent.graph import (
    EXPERIMENT_NAME,
    GATEWAY_ROUTE,
    build_orchestrator_graph,
    invoke_config,
)
import pytest

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(task: str) -> dict[str, object]:
    checkpointer = InMemorySaver()
    graph = build_orchestrator_graph(checkpointer=checkpointer)
    config = invoke_config(task)
    result = graph.invoke(
        {"task": task, "analysis": "", "subtasks": [], "worker_results": [], "synthesis": ""},
        config=config,
    )
    return {"subtask_count": len(result["subtasks"]), "synthesis": result["synthesis"]}


@scorer
def meets_minimum_subtask_count(outputs: dict[str, object], expectations: dict[str, int]) -> bool:
    """Did the orchestrator decompose the task into at least the expected minimum subtasks?"""
    return outputs["subtask_count"] >= expectations["min_subtasks"]


@pytest.mark.eval
def test_orchestrator_workers_agent_eval_suite() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name="orchestrator-workers-agent-eval"):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=_predict_fn,
            scorers=[
                meets_minimum_subtask_count,
                Guidelines(
                    name="coherent_synthesis",
                    guidelines=load_judge_guidelines(
                        "orchestrator-workers-agent-coherent_synthesis"
                    ),
                    model=_JUDGE_MODEL_URI,
                ),
                Safety(model=_JUDGE_MODEL_URI),  # type: ignore[no-untyped-call]
            ],
        )

    # Threshold, not a hardcoded 100% — LLM-judged scores are noisy by nature, and orchestrator
    # decomposition choices vary run to run. Tune once real eval history exists in MLflow (see
    # the MLflow UI's evaluation-runs tab for this experiment).
    assert results.metrics["meets_minimum_subtask_count/mean"] >= 0.7
    assert results.metrics["coherent_synthesis/mean"] >= 0.7


@pytest.mark.regression
def test_orchestrator_workers_agent_regression() -> None:
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

    with mlflow.start_run(run_name="orchestrator-workers-agent-regression"):
        results = mlflow.genai.evaluate(
            data=records,
            predict_fn=_predict_fn,
            scorers=[meets_minimum_subtask_count],
        )

    assert results.metrics["meets_minimum_subtask_count/mean"] >= 0.95
