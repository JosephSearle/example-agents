"""MLflow GenAI eval suite for the network/mesh workflow.

Calls a real model, so this is gated out of the default CI run (see `.github/workflows/eval.yml`
and the `eval` marker) — run explicitly with `pytest -m eval`. Results land in MLflow as a run
under this agent's own `network_mesh_agent.EXPERIMENT_NAME` experiment.

The dataset lives in MLflow's dataset registry, seeded from
`packages/mlflow-server/datasets/network-mesh-agent.jsonl` via `make provision-datasets`. Unlike
orchestrator-workers-agent's `min_subtasks` lower bound, this pattern has no single per-record
expectation to check against — whether the mesh takes the direct researcher->writer path or loops
through the critic first is itself a runtime routing decision, not something to assert on per
input. So this suite scores quality only: whether the final answer is grounded in the mesh's own
research/critique transcript.
"""

from __future__ import annotations

import os

from agents_common import configure_mlflow
from agents_common.config import get_settings
from agents_common.judges import load_judge_guidelines
from langgraph.checkpoint.memory import InMemorySaver
import mlflow
from mlflow.genai.datasets import get_dataset
from mlflow.genai.scorers import Guidelines
from network_mesh_agent.graph import EXPERIMENT_NAME, GATEWAY_ROUTE, build_mesh_graph, invoke_config
import pytest

pytestmark = pytest.mark.eval

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(task: str) -> dict[str, object]:
    checkpointer = InMemorySaver()
    graph = build_mesh_graph(checkpointer=checkpointer)
    config = invoke_config(task)
    result = graph.invoke(
        {
            "task": task,
            "messages": [],
            "needs_critique": False,
            "needs_more_research": False,
            "research_rounds": 0,
            "final_answer": "",
        },
        config=config,
    )
    return {"final_answer": result["final_answer"], "transcript": "\n\n".join(result["messages"])}


def test_network_mesh_agent_eval_suite() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name="network-mesh-agent-eval"):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=_predict_fn,
            scorers=[
                Guidelines(
                    name="grounded_in_research",
                    guidelines=load_judge_guidelines("network-mesh-agent-grounded_in_research"),
                    model=_JUDGE_MODEL_URI,
                ),
            ],
        )

    # Threshold, not a hardcoded 100% — LLM-judged scores are noisy by nature. Tune once real
    # eval history exists in MLflow (see the MLflow UI's evaluation-runs tab for this experiment).
    assert results.metrics["grounded_in_research/mean"] >= 0.7
