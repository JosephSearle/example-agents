"""MLflow GenAI eval suite for the supervisor workflow.

Calls a real model, so this is gated out of the default CI run (see `.github/workflows/eval.yml`
and the `eval` marker) — run explicitly with `pytest -m eval`. Results land in MLflow as a run
under this agent's own `supervisor_agent.EXPERIMENT_NAME` experiment.

The dataset lives in MLflow's dataset registry, seeded from
`packages/mlflow-server/datasets/supervisor-agent.jsonl` via `make provision-datasets`. Each
record's `expectations.expected_delegates` is the set of delegate tools the supervisor should
have called (`delegate_to_math`, `delegate_to_text`, or both) — scored as a subset check against
the tool calls actually observed in the supervisor's own message trajectory, rather than an exact
match: a reasonable supervisor may legitimately re-delegate or add a follow-up call the dataset
doesn't anticipate, so this only asserts the *required* delegations happened, same spirit as
routing-agent's exact-match `correct_category` but loosened for this pattern's messier trajectory.
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
import pytest
from supervisor_agent.graph import (
    EXPERIMENT_NAME,
    GATEWAY_ROUTE,
    build_supervisor,
    invoke_config,
)

pytestmark = pytest.mark.eval

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"

_DELEGATE_TOOL_NAMES = {"delegate_to_math", "delegate_to_text"}


def _predict_fn(task: str) -> dict[str, object]:
    checkpointer = InMemorySaver()
    supervisor = build_supervisor(checkpointer=checkpointer)
    config = invoke_config(task)
    result = supervisor.invoke({"messages": [{"role": "user", "content": task}]}, config=config)
    delegated = sorted(
        {
            tool_call["name"]
            for message in result["messages"]
            for tool_call in getattr(message, "tool_calls", [])
            if tool_call["name"] in _DELEGATE_TOOL_NAMES
        }
    )
    return {"delegated": delegated, "answer": str(result["messages"][-1].content)}


@scorer
def delegated_to_the_required_sub_agents(
    outputs: dict[str, object], expectations: dict[str, list[str]]
) -> bool:
    """Did the supervisor call at least every delegate tool the task genuinely required?"""
    delegated = set(outputs["delegated"])  # type: ignore[arg-type]
    return set(expectations["expected_delegates"]).issubset(delegated)


def test_supervisor_agent_eval_suite() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name="supervisor-agent-eval"):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=_predict_fn,
            scorers=[
                delegated_to_the_required_sub_agents,
                Guidelines(
                    name="delegates_appropriately",
                    guidelines=load_judge_guidelines("supervisor-agent-delegates_appropriately"),
                    model=_JUDGE_MODEL_URI,
                ),
            ],
        )

    # Threshold, not a hardcoded 100% — LLM-judged scores are noisy by nature, and delegation
    # choices vary run to run. Tune once real eval history exists in MLflow (see the MLflow UI's
    # evaluation-runs tab for this experiment).
    assert results.metrics["delegated_to_the_required_sub_agents/mean"] >= 0.7
    assert results.metrics["delegates_appropriately/mean"] >= 0.7
