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
from agents_common.judges import (
    Safety,
    load_judge_guidelines,
    pass_at_k,
    regression_subset,
    tool_calls_include,
)
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

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(question: str) -> dict[str, object]:
    prompt_version = load_system_prompt_version()
    checkpointer = InMemorySaver()
    agent = build_agent(checkpointer=checkpointer, system_prompt=prompt_text(prompt_version))
    config = invoke_config(str(uuid.uuid4()))
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)  # type: ignore[arg-type]
    link_prompt_to_trace(prompt_version, mlflow.get_last_active_trace_id())
    tool_calls = sorted(
        {
            tool_call["name"]
            for message in result["messages"]
            for tool_call in getattr(message, "tool_calls", [])
        }
    )
    return {"answer": extract_response(result).answer, "tool_calls": tool_calls}


_calculator_called = tool_calls_include(
    "calls_calculator_when_needed", trajectory_key="tool_calls", expected_key="expected_tool_calls"
)


@pytest.mark.eval
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
                _calculator_called,
                Safety(model=_JUDGE_MODEL_URI),  # type: ignore[no-untyped-call]
            ],
        )

    # Threshold, not a hardcoded 100% — LLM-judged scores are noisy by nature. Tune once real
    # eval history exists in MLflow.
    assert results.metrics["correctness/mean"] >= 0.8


@pytest.mark.eval
def test_react_agent_calculator_pass_at_k() -> None:
    """Known open issue (not a wiring bug — dataset loading, structured-output extraction, and
    judge routing are all confirmed working): gpt-oss-120b intermittently fails to call
    `calculator` at all, returning `{}` or restating the question instead of computing an answer.
    `Correctness` scores on the full suite bounced between 0.33 and 0.67 across runs before this
    was isolated to the single flaky question below.

    Uses `agents_common.judges.pass_at_k` — the one worked example of pass@k/pass^k in this repo
    (see docs/decisions/0002-eval-taxonomy.md). Copy this pattern into another pattern's suite
    only if it has similarly *observed* non-determinism; it isn't applied blanket.
    """
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)
    configure_mlflow(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="react-agent-calculator-pass-at-k"):
        rate = pass_at_k(
            _predict_fn,
            "What is 47 * 12?",
            k=5,
            is_success=lambda out: "calculator" in out["tool_calls"],  # type: ignore[operator]
        )
        mlflow.log_metric("calculator_pass_at_5", rate)

    # Documented flake rate is ~0.33-0.67; assert the loose floor so this stays a real signal
    # (not silently disabled) without flapping CI red on every run.
    assert rate >= 0.2


@pytest.mark.regression
def test_react_agent_regression() -> None:
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

    with mlflow.start_run(run_name="react-agent-regression"):
        results = mlflow.genai.evaluate(
            data=records,
            predict_fn=_predict_fn,
            scorers=[_calculator_called],
        )

    assert results.metrics["calls_calculator_when_needed/mean"] >= 0.95
