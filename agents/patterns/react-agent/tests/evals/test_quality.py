"""MLflow GenAI eval suite for the ReAct agent.

Calls a real model, so this is gated out of the default CI run (see `.github/workflows/eval.yml`
and the `eval` marker) — run explicitly with `pytest -m eval`, or via the nightly/label-triggered
GitHub Actions workflow. Results land in MLflow as a run under this agent's own
`react_agent.EXPERIMENT_NAME` experiment, so eval quality is trended over time rather than
being a one-off pass/fail in CI logs.

The dataset itself lives in MLflow's dataset registry, not in this package — seeded from
`packages/mlflow-server/datasets/react-agent.jsonl` via `make provision-datasets` (run once after
`make up`, and again after editing that JSONL). `EXPERIMENT_NAME` doubles as both the MLflow
experiment name and the dataset name, per that provisioning script's naming convention.

The LLM-judge scorers (`Correctness`, `Guidelines`) default to calling real OpenAI
(`openai:/gpt-4.1-mini`, requiring `OPENAI_API_KEY`) when no `model=` is given — there is no
Databricks-managed judge model to fall back to here. `_JUDGE_MODEL_URI` points them at our own
gateway route instead: `mlflow.genai.judges`' "openai" provider reads `OPENAI_API_KEY` /
`OPENAI_API_BASE` directly from the environment (the same mechanism a raw `ChatOpenAI` client
uses), so pointing `OPENAI_API_BASE` at `Settings.mlflow_gateway_base_url` routes judge calls
through the same self-hosted model the agent itself uses — confirmed working directly against
`mlflow.genai.judges.is_correct`.
"""

from __future__ import annotations

import os
import uuid

from agents_common import configure_mlflow
from agents_common.config import get_settings
from agents_common.judges import load_judge_guidelines
from langgraph.checkpoint.memory import InMemorySaver
import mlflow
from mlflow.genai.datasets import get_dataset
from mlflow.genai.scorers import Correctness, Guidelines
import pytest
from react_agent.graph import (
    EXPERIMENT_NAME,
    GATEWAY_ROUTE,
    build_agent,
    extract_response,
    invoke_config,
    link_prompt_to_trace,
    load_system_prompt_version,
    prompt_text,
)

pytestmark = pytest.mark.eval

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(question: str) -> str:
    prompt_version = load_system_prompt_version()
    checkpointer = InMemorySaver()
    agent = build_agent(checkpointer=checkpointer, system_prompt=prompt_text(prompt_version))
    config = invoke_config(str(uuid.uuid4()))
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)  # type: ignore[arg-type]
    link_prompt_to_trace(prompt_version, mlflow.get_last_active_trace_id())
    return extract_response(result).answer


def test_react_agent_eval_suite() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name="react-agent-eval"):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=_predict_fn,
            scorers=[
                Correctness(model=_JUDGE_MODEL_URI),
                Guidelines(
                    name="concise_answer",
                    guidelines=load_judge_guidelines("react-agent-concise_answer"),
                    model=_JUDGE_MODEL_URI,
                ),
            ],
        )

    # Threshold, not a hardcoded 100% — LLM-judged scores are noisy by nature. Tune once real
    # eval history exists in MLflow.
    #
    # Known open issue (not a wiring bug — dataset loading, structured-output extraction, and
    # judge routing are all confirmed working): gpt-oss-120b intermittently fails to call
    # `calculator` at all, returning `{}` or restating the question instead of computing an
    # answer. Scores have bounced between 0.33 and 0.67 across runs on this 3-question dataset.
    # Left failing deliberately as a signal — see the MLflow UI's evaluation-runs tab for this
    # experiment (MLFLOW_TRACKING_URI + "/#/experiments/<id>/evaluation-runs") for judge
    # rationale on each failure.
    assert results.metrics["correctness/mean"] >= 0.8
