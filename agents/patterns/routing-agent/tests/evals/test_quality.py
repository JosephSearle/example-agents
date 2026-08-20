"""MLflow GenAI eval suite for the routing workflow.

Calls a real model, so this is gated out of the default CI run (see `.github/workflows/eval.yml`
and the `eval` marker) — run explicitly with `pytest -m eval`. Results land in MLflow as a run
under this agent's own `routing_agent.EXPERIMENT_NAME` experiment.

The dataset lives in MLflow's dataset registry, seeded from
`packages/mlflow-server/datasets/routing-agent.jsonl` via `make provision-datasets`. Each record's
`expectations.expected_category` is the ground-truth route the classifier should have picked —
scored directly (exact match) rather than by an LLM judge, since routing correctness is a
deterministic string comparison, not a subjective quality judgment like
`prompt_chaining_agent`'s prose-quality eval.

See `react_agent`'s eval suite for why a second, judge-based scorer (here, on response relevance)
points the judge at our own gateway route via `OPENAI_API_KEY`/`OPENAI_API_BASE` rather than a
managed judge model.
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
from routing_agent.graph import EXPERIMENT_NAME, GATEWAY_ROUTE, build_router, invoke_config

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(message: str) -> dict[str, str]:
    checkpointer = InMemorySaver()
    router = build_router(checkpointer=checkpointer)
    config = invoke_config(message)
    result = router.invoke({"message": message, "category": "", "response": ""}, config=config)
    return {"category": result["category"], "response": result["response"]}


@scorer
def correct_category(outputs: dict[str, str], expectations: dict[str, str]) -> bool:
    """Exact-match check: did the classifier pick the ground-truth category for this ticket?"""
    return outputs["category"] == expectations["expected_category"]


@pytest.mark.eval
def test_routing_agent_eval_suite() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name="routing-agent-eval"):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=_predict_fn,
            scorers=[
                correct_category,
                Guidelines(
                    name="relevant_response",
                    guidelines=load_judge_guidelines("routing-agent-relevant_response"),
                    model=_JUDGE_MODEL_URI,
                ),
                Safety(model=_JUDGE_MODEL_URI),  # type: ignore[no-untyped-call]
            ],
        )

    # Threshold, not a hardcoded 100% — LLM-judged scores are noisy by nature, and even the exact
    # match category scorer is allowed some slack for classifier variance. Tune once real eval
    # history exists in MLflow (see the MLflow UI's evaluation-runs tab for this experiment).
    assert results.metrics["correct_category/mean"] >= 0.7
    assert results.metrics["relevant_response/mean"] >= 0.7


@pytest.mark.regression
def test_routing_agent_regression() -> None:
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

    with mlflow.start_run(run_name="routing-agent-regression"):
        results = mlflow.genai.evaluate(
            data=records,
            predict_fn=_predict_fn,
            scorers=[correct_category],
        )

    assert results.metrics["correct_category/mean"] >= 0.95
