"""MLflow GenAI eval suite for the map-reduce workflow.

Calls a real model, so this is gated out of the default CI run (see `.github/workflows/eval.yml`
and the `eval` marker) — run explicitly with `pytest -m eval`. Results land in MLflow as a run
under this agent's own `map_reduce_agent.EXPERIMENT_NAME` experiment.

The dataset lives in MLflow's dataset registry, seeded from
`packages/mlflow-server/datasets/map-reduce-agent.jsonl` via `make provision-datasets`. Each
record's `expectations.expected_joke_count` is the ground-truth number of jokes the dynamic
fan-out should have produced — scored directly (exact match against `len(topics)`) rather than by
an LLM judge, same convention as routing-agent's `correct_category` scorer. This is the one
property genuinely specific to map-reduce (vs. any other pattern): fan-out width tracks input
size at runtime, so this scorer is what actually exercises that, independent of joke quality.
"""

from __future__ import annotations

import os

from agents_common import configure_mlflow
from agents_common.config import get_settings
from agents_common.judges import Safety, load_judge_guidelines, regression_subset
from langgraph.checkpoint.memory import InMemorySaver
from map_reduce_agent.graph import (
    EXPERIMENT_NAME,
    GATEWAY_ROUTE,
    build_map_reduce_graph,
    invoke_config,
)
import mlflow
from mlflow.genai.datasets import get_dataset
from mlflow.genai.scorers import Guidelines, scorer
import pytest

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(topics: list[str]) -> dict[str, object]:
    checkpointer = InMemorySaver()
    graph = build_map_reduce_graph(checkpointer=checkpointer)
    config = invoke_config(str(topics))
    result = graph.invoke({"topics": topics, "jokes": [], "summary": ""}, config=config)
    return {"joke_count": len(result["jokes"]), "summary": result["summary"]}


@scorer
def correct_joke_count(outputs: dict[str, object], expectations: dict[str, int]) -> bool:
    """Exact-match check: did dynamic fan-out produce exactly one joke per input topic?"""
    return outputs["joke_count"] == expectations["expected_joke_count"]


@pytest.mark.eval
def test_map_reduce_agent_eval_suite() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name="map-reduce-agent-eval"):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=_predict_fn,
            scorers=[
                correct_joke_count,
                Guidelines(
                    name="relevant_jokes",
                    guidelines=load_judge_guidelines("map-reduce-agent-relevant_jokes"),
                    model=_JUDGE_MODEL_URI,
                ),
                Safety(model=_JUDGE_MODEL_URI),  # type: ignore[no-untyped-call]
            ],
        )

    # Threshold, not a hardcoded 100% — LLM-judged scores are noisy by nature, and even the exact
    # match joke-count scorer is allowed some slack. Tune once real eval history exists in MLflow
    # (see the MLflow UI's evaluation-runs tab for this experiment).
    assert results.metrics["correct_joke_count/mean"] >= 0.7
    assert results.metrics["relevant_jokes/mean"] >= 0.7


@pytest.mark.regression
def test_map_reduce_agent_regression() -> None:
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

    with mlflow.start_run(run_name="map-reduce-agent-regression"):
        results = mlflow.genai.evaluate(
            data=records,
            predict_fn=_predict_fn,
            scorers=[correct_joke_count],
        )

    assert results.metrics["correct_joke_count/mean"] >= 0.95
