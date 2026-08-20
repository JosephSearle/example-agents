"""MLflow GenAI eval suite for the prompt chaining workflow.

Calls a real model, so this is gated out of the default CI run (see `.github/workflows/eval.yml`
and the `eval` marker) — run explicitly with `pytest -m eval`. Results land in MLflow as a run
under this agent's own `prompt_chaining_agent.EXPERIMENT_NAME` experiment.

The dataset lives in MLflow's dataset registry, seeded from
`packages/mlflow-server/datasets/prompt-chaining-agent.jsonl` via `make provision-datasets`.

See `react_agent`'s eval suite for why `_JUDGE_MODEL_URI` points the judge at our own gateway
route via `OPENAI_API_KEY`/`OPENAI_API_BASE` rather than a managed judge model.
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
from prompt_chaining_agent.graph import EXPERIMENT_NAME, GATEWAY_ROUTE, build_chain, invoke_config
import pytest

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(topic: str) -> str:
    checkpointer = InMemorySaver()
    chain = build_chain(checkpointer=checkpointer)
    config = invoke_config(topic)
    result = chain.invoke({"topic": topic, "outline": "", "draft": "", "final": ""}, config=config)
    return result["final"]


@pytest.mark.eval
def test_prompt_chaining_agent_eval_suite() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name="prompt-chaining-agent-eval"):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=_predict_fn,
            scorers=[
                Guidelines(
                    name="well_formed_prose",
                    guidelines=load_judge_guidelines("prompt-chaining-agent-well_formed_prose"),
                    model=_JUDGE_MODEL_URI,
                ),
                Safety(model=_JUDGE_MODEL_URI),  # type: ignore[no-untyped-call]
            ],
        )

    # Threshold, not a hardcoded 100% — LLM-judged scores are noisy by nature. Tune once real
    # eval history exists in MLflow (see the MLflow UI's evaluation-runs tab for this
    # experiment).
    assert results.metrics["well_formed_prose/mean"] >= 0.7


@scorer
def non_empty_final(outputs: str) -> bool:
    """Fallback code-based check for the regression suite: the chain's step count (outline, draft,
    polish) is fixed in code rather than a runtime signal tied to the input (contrast
    map-reduce-agent's `correct_joke_count`), so there's no deterministic trajectory scorer to
    reuse here — this just confirms the chain actually produced a final, non-empty answer.
    """
    return len(str(outputs)) > 0


@pytest.mark.regression
def test_prompt_chaining_agent_regression() -> None:
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

    with mlflow.start_run(run_name="prompt-chaining-agent-regression"):
        results = mlflow.genai.evaluate(
            data=records,
            predict_fn=_predict_fn,
            scorers=[non_empty_final],
        )

    assert results.metrics["non_empty_final/mean"] >= 0.95
