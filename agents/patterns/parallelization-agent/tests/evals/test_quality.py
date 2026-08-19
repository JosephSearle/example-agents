"""MLflow GenAI eval suite for the parallelization (sectioning) workflow.

Calls a real model, so this is gated out of the default CI run (see `.github/workflows/eval.yml`
and the `eval` marker) — run explicitly with `pytest -m eval`. Results land in MLflow as a run
under this agent's own `parallelization_agent.EXPERIMENT_NAME` experiment.

The dataset lives in MLflow's dataset registry, seeded from
`packages/mlflow-server/datasets/parallelization-agent.jsonl` via `make provision-datasets`. Each
record's `expectations.expected_severity` is the ground-truth severity the `assess_severity`
section should have picked — scored directly (exact match) rather than by an LLM judge, same
convention as routing-agent's `correct_category` scorer.
"""

from __future__ import annotations

import os

from agents_common import configure_mlflow
from agents_common.config import get_settings
from agents_common.judges import load_judge_guidelines
from langgraph.checkpoint.memory import InMemorySaver
import mlflow
from mlflow.genai.datasets import get_dataset
from mlflow.genai.scorers import Guidelines, scorer
from parallelization_agent.graph import (
    EXPERIMENT_NAME,
    GATEWAY_ROUTE,
    build_sectioning_graph,
    invoke_config,
)
import pytest

pytestmark = pytest.mark.eval

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(incident_text: str) -> dict[str, object]:
    checkpointer = InMemorySaver()
    graph = build_sectioning_graph(checkpointer=checkpointer)
    config = invoke_config(incident_text)
    result = graph.invoke(
        {
            "incident_text": incident_text,
            "summary": "",
            "severity": "",
            "action_items": [],
            "report": "",
        },
        config=config,
    )
    return {"severity": result["severity"], "report": result["report"]}


@scorer
def correct_severity(outputs: dict[str, object], expectations: dict[str, str]) -> bool:
    """Exact-match check: did `assess_severity` pick the ground-truth severity for this incident?"""
    return outputs["severity"] == expectations["expected_severity"]


def test_parallelization_agent_eval_suite() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name="parallelization-agent-eval"):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=_predict_fn,
            scorers=[
                correct_severity,
                Guidelines(
                    name="coherent_report",
                    guidelines=load_judge_guidelines("parallelization-agent-coherent_report"),
                    model=_JUDGE_MODEL_URI,
                ),
            ],
        )

    # Threshold, not a hardcoded 100% — LLM-judged scores are noisy by nature, and even the exact
    # match severity scorer is allowed some slack for classifier variance. Tune once real eval
    # history exists in MLflow (see the MLflow UI's evaluation-runs tab for this experiment).
    assert results.metrics["correct_severity/mean"] >= 0.7
    assert results.metrics["coherent_report/mean"] >= 0.7
