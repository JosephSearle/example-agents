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
from langgraph.checkpoint.memory import InMemorySaver
import mlflow
from mlflow.genai.datasets import get_dataset
from mlflow.genai.scorers import Guidelines
from prompt_chaining_agent.graph import EXPERIMENT_NAME, GATEWAY_ROUTE, build_chain, invoke_config
import pytest

pytestmark = pytest.mark.eval

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(topic: str) -> str:
    checkpointer = InMemorySaver()
    chain = build_chain(checkpointer=checkpointer)
    config = invoke_config(topic)
    result = chain.invoke({"topic": topic, "outline": "", "draft": "", "final": ""}, config=config)
    return result["final"]


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
                    guidelines=(
                        "The response must be flowing prose paragraphs, not a bulleted or "
                        "numbered outline."
                    ),
                    model=_JUDGE_MODEL_URI,
                ),
            ],
        )

    # Threshold, not a hardcoded 100% — LLM-judged scores are noisy by nature. Tune once real
    # eval history exists in MLflow (see the MLflow UI's evaluation-runs tab for this
    # experiment).
    assert results.metrics["well_formed_prose/mean"] >= 0.7
