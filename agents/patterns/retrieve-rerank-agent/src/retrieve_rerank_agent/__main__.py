"""CLI entrypoint: `uv run retrieve-rerank-agent "<question>"`."""

from __future__ import annotations

import sys
import uuid

from agents_common import configure_mlflow, get_checkpointer
import mlflow

from retrieve_rerank_agent.graph import (
    EXPERIMENT_NAME,
    build_rag_graph,
    invoke_config,
    link_prompt_to_trace,
    load_rag_prompt_version,
    prompt_text,
)

_MIN_ARGC = 2


def main() -> None:
    """Retrieve a wide candidate set from Milvus, rerank it, generate a grounded answer, and print it."""
    if len(sys.argv) < _MIN_ARGC:
        print('Usage: retrieve-rerank-agent "<question>"', file=sys.stderr)
        raise SystemExit(1)

    question = sys.argv[1]
    configure_mlflow(EXPERIMENT_NAME)

    prompt_version = load_rag_prompt_version()

    with get_checkpointer() as checkpointer:
        graph = build_rag_graph(checkpointer=checkpointer, rag_prompt=prompt_text(prompt_version))
        config = invoke_config(str(uuid.uuid4()))
        result = graph.invoke(  # type: ignore[call-overload]
            {"question": question, "candidate_chunks": [], "reranked_chunks": [], "answer": ""},
            config=config,
        )
        link_prompt_to_trace(prompt_version, mlflow.get_last_active_trace_id())
        print(result["answer"])


if __name__ == "__main__":
    main()
