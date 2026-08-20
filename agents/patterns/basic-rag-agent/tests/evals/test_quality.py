"""MLflow GenAI eval suite for the basic RAG workflow.

Calls a real model and real Milvus retrieval, so this is gated out of the default CI run (see
`.github/workflows/eval.yml` and the `eval` marker) — run explicitly with `pytest -m eval`.
Results land in MLflow as a run under this agent's own `basic_rag_agent.EXPERIMENT_NAME`
experiment.

The dataset lives in MLflow's dataset registry, seeded from
`packages/mlflow-server/datasets/basic-rag-agent.jsonl` via `make provision-datasets` — questions
answerable from the seed corpus at `packages/milvus/collections/basic-rag-agent.jsonl` (chunks of
this repo's own docs). Scored only on `grounded_in_context` (no exact-match scorer, unlike
routing-agent's `correct_category`): there's no single ground-truth answer string to match, only
whether the answer reflects what was actually retrieved.
"""

from __future__ import annotations

import os

from agents_common import configure_mlflow
from agents_common.config import get_settings
from agents_common.judges import load_judge_guidelines
from basic_rag_agent.graph import EXPERIMENT_NAME, GATEWAY_ROUTE, build_rag_graph, invoke_config
from langgraph.checkpoint.memory import InMemorySaver
import mlflow
from mlflow.genai.datasets import get_dataset
from mlflow.genai.scorers import Guidelines
import pytest

pytestmark = pytest.mark.eval

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(question: str) -> dict[str, object]:
    checkpointer = InMemorySaver()
    graph = build_rag_graph(checkpointer=checkpointer)
    config = invoke_config(question)
    result = graph.invoke(
        {"question": question, "retrieved_chunks": [], "answer": ""}, config=config
    )
    return {"answer": result["answer"], "retrieved_chunks": result["retrieved_chunks"]}


def test_basic_rag_agent_eval_suite() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name="basic-rag-agent-eval"):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=_predict_fn,
            scorers=[
                Guidelines(
                    name="grounded_in_context",
                    guidelines=load_judge_guidelines("basic-rag-agent-grounded_in_context"),
                    model=_JUDGE_MODEL_URI,
                ),
            ],
        )

    # Threshold, not a hardcoded 100% — LLM-judged scores are noisy by nature. Tune once real
    # eval history exists in MLflow (see the MLflow UI's evaluation-runs tab for this experiment).
    assert results.metrics["grounded_in_context/mean"] >= 0.7
