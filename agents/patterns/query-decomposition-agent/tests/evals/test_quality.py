"""MLflow GenAI eval suite for the Query Decomposition RAG workflow.

Calls a real model and real Milvus retrieval, so this is gated out of the default CI run — run
explicitly with `pytest -m eval`. Results land in MLflow as a run under this agent's own
`query_decomposition_agent.EXPERIMENT_NAME` experiment.

The dataset lives in MLflow's dataset registry, seeded from
`packages/mlflow-server/datasets/query-decomposition-agent.jsonl` via `make provision-datasets` —
genuinely multi-part questions, hand-curated (not reused from basic-rag-agent's single-fact
dataset), since this pattern's whole point is decomposition.
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
from query_decomposition_agent.graph import (
    EXPERIMENT_NAME,
    GATEWAY_ROUTE,
    build_rag_graph,
    invoke_config,
)

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"


def _predict_fn(question: str) -> dict[str, object]:
    checkpointer = InMemorySaver()
    graph = build_rag_graph(checkpointer=checkpointer)
    config = invoke_config(question)
    result = graph.invoke(
        {"question": question, "sub_questions": [], "sub_answers": [], "answer": ""},
        config=config,
    )
    return {
        "answer": result["answer"],
        "sub_questions": result["sub_questions"],
        "sub_answers": result["sub_answers"],
    }


@scorer
def decomposed_into_multiple_parts(
    outputs: dict[str, object], expectations: dict[str, object]
) -> bool:
    """Structural check on `decompose`: for a genuinely multi-part question, did it actually
    split rather than pass the question through unchanged (sub-question wording itself isn't
    exact-matchable, but "did it decompose at all" is).
    """
    sub_questions = outputs["sub_questions"]
    assert isinstance(sub_questions, list)
    return len(sub_questions) >= expectations["expected_min_sub_questions"]


@pytest.mark.eval
def test_query_decomposition_agent_eval_suite() -> None:
    settings = get_settings()
    os.environ.setdefault("OPENAI_API_KEY", settings.mlflow_tracking_token or "unused")
    os.environ.setdefault("OPENAI_API_BASE", settings.mlflow_gateway_base_url)

    configure_mlflow(EXPERIMENT_NAME)
    dataset = get_dataset(name=EXPERIMENT_NAME)

    with mlflow.start_run(run_name="query-decomposition-agent-eval"):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=_predict_fn,
            scorers=[
                decomposed_into_multiple_parts,
                Guidelines(
                    name="grounded_in_context",
                    guidelines=load_judge_guidelines(
                        "query-decomposition-agent-grounded_in_context"
                    ),
                    model=_JUDGE_MODEL_URI,
                ),
                Guidelines(
                    name="addresses_original_question",
                    guidelines=load_judge_guidelines(
                        "query-decomposition-agent-addresses_original_question"
                    ),
                    model=_JUDGE_MODEL_URI,
                ),
                Safety(model=_JUDGE_MODEL_URI),  # type: ignore[no-untyped-call]
            ],
        )

    assert results.metrics["grounded_in_context/mean"] >= 0.7
    assert results.metrics["addresses_original_question/mean"] >= 0.7
    assert results.metrics["decomposed_into_multiple_parts/mean"] >= 0.7


@pytest.mark.regression
def test_query_decomposition_agent_regression() -> None:
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

    with mlflow.start_run(run_name="query-decomposition-agent-regression"):
        results = mlflow.genai.evaluate(
            data=records,
            predict_fn=_predict_fn,
            scorers=[decomposed_into_multiple_parts],
        )

    assert results.metrics["decomposed_into_multiple_parts/mean"] >= 0.95
