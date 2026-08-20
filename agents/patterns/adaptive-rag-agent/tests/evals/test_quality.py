"""MLflow GenAI eval suite for the Adaptive RAG workflow.

Calls a real model and real Milvus retrieval, so this is gated out of the default CI run — run
explicitly with `pytest -m eval`. Results land in MLflow as a run under this agent's own
`adaptive_rag_agent.EXPERIMENT_NAME` experiment.

The dataset lives in MLflow's dataset registry, seeded from
`packages/mlflow-server/datasets/adaptive-rag-agent.jsonl` via `make provision-datasets` —
deliberately spans all three complexity tiers (general-knowledge, single-fact, multi-part
questions), since router miscalibration is this pattern's headline risk.
"""

from __future__ import annotations

import os

from adaptive_rag_agent.graph import EXPERIMENT_NAME, GATEWAY_ROUTE, build_rag_graph, invoke_config
from agents_common import configure_mlflow
from agents_common.config import get_settings
from agents_common.judges import Safety, load_judge_guidelines, regression_subset
from langgraph.checkpoint.memory import InMemorySaver
import mlflow
from mlflow.genai.datasets import get_dataset
from mlflow.genai.scorers import Guidelines, scorer
import pytest

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(question: str) -> dict[str, object]:
    checkpointer = InMemorySaver()
    graph = build_rag_graph(checkpointer=checkpointer)
    config = invoke_config(question)
    result = graph.invoke(
        {
            "question": question,
            "complexity": "",
            "documents": [],
            "documents_sufficient": False,
            "retry_count": 0,
            "sub_questions": [],
            "sub_answers": [],
            "answer": "",
        },
        config=config,
    )
    return {"answer": result["answer"], "complexity": result["complexity"]}


@scorer
def correct_route(outputs: dict[str, object], expectations: dict[str, object]) -> bool:
    """Exact-match check on `route_by_complexity`: did the conditional entry point pick the
    ground-truth tier, rather than the LLM-judged `routed_appropriately`'s subjective read of
    whether the *final answer* looks appropriate for whatever tier got picked?
    """
    return outputs["complexity"] == expectations["expected_complexity"]


@pytest.mark.eval
def test_adaptive_rag_agent_eval_suite() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name="adaptive-rag-agent-eval"):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=_predict_fn,
            scorers=[
                correct_route,
                Guidelines(
                    name="grounded_in_context",
                    guidelines=load_judge_guidelines("adaptive-rag-agent-grounded_in_context"),
                    model=_JUDGE_MODEL_URI,
                ),
                Guidelines(
                    name="routed_appropriately",
                    guidelines=load_judge_guidelines("adaptive-rag-agent-routed_appropriately"),
                    model=_JUDGE_MODEL_URI,
                ),
                Safety(model=_JUDGE_MODEL_URI),  # type: ignore[no-untyped-call]
            ],
        )

    assert results.metrics["grounded_in_context/mean"] >= 0.7
    assert results.metrics["routed_appropriately/mean"] >= 0.7
    assert results.metrics["correct_route/mean"] >= 0.7


@pytest.mark.regression
def test_adaptive_rag_agent_regression() -> None:
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

    with mlflow.start_run(run_name="adaptive-rag-agent-regression"):
        results = mlflow.genai.evaluate(
            data=records,
            predict_fn=_predict_fn,
            scorers=[correct_route],
        )

    assert results.metrics["correct_route/mean"] >= 0.95
