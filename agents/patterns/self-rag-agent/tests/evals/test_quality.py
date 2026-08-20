"""MLflow GenAI eval suite for the Self-RAG workflow.

Calls a real model and real Milvus retrieval, so this is gated out of the default CI run — run
explicitly with `pytest -m eval`. Results land in MLflow as a run under this agent's own
`self_rag_agent.EXPERIMENT_NAME` experiment.

The dataset lives in MLflow's dataset registry, seeded from
`packages/mlflow-server/datasets/self-rag-agent.jsonl` via `make provision-datasets`.
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
import pytest
from self_rag_agent.graph import EXPERIMENT_NAME, GATEWAY_ROUTE, build_rag_graph, invoke_config

pytestmark = pytest.mark.eval

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(question: str) -> dict[str, object]:
    checkpointer = InMemorySaver()
    graph = build_rag_graph(checkpointer=checkpointer)
    config = invoke_config(question)
    result = graph.invoke(
        {
            "question": question,
            "original_question": question,
            "documents": [],
            "documents_sufficient": False,
            "retry_count": 0,
            "answer": "",
            "grounded": False,
            "useful": False,
            "regenerate_count": 0,
        },
        config=config,
    )
    return {"answer": result["answer"], "documents": result["documents"]}


def test_self_rag_agent_eval_suite() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name="self-rag-agent-eval"):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=_predict_fn,
            scorers=[
                Guidelines(
                    name="grounded_in_context",
                    guidelines=load_judge_guidelines("self-rag-agent-grounded_in_context"),
                    model=_JUDGE_MODEL_URI,
                ),
            ],
        )

    assert results.metrics["grounded_in_context/mean"] >= 0.7
