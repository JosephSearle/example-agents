"""CLI entrypoint: `uv run adaptive-rag-agent "<question>"`."""

from __future__ import annotations

import sys
import uuid

from agents_common import configure_mlflow, get_checkpointer
import mlflow

from adaptive_rag_agent.graph import (
    EXPERIMENT_NAME,
    PROMPT_NAMES,
    build_rag_graph,
    invoke_config,
    link_prompts_to_trace,
    load_rag_prompt_version,
    prompt_text,
)

_MIN_ARGC = 2


def main() -> None:
    """Route a question by complexity to a no-retrieval, single-step, or multi-step branch, and print the answer."""
    if len(sys.argv) < _MIN_ARGC:
        print('Usage: adaptive-rag-agent "<question>"', file=sys.stderr)
        raise SystemExit(1)

    question = sys.argv[1]
    configure_mlflow(EXPERIMENT_NAME)

    prompt_versions = {step: load_rag_prompt_version(step) for step in PROMPT_NAMES}
    prompts = {step: prompt_text(version) for step, version in prompt_versions.items()}

    with get_checkpointer() as checkpointer:
        graph = build_rag_graph(checkpointer=checkpointer, prompts=prompts)
        config = invoke_config(str(uuid.uuid4()))
        result = graph.invoke(  # type: ignore[call-overload]
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
        link_prompts_to_trace(prompt_versions, mlflow.get_last_active_trace_id())
        print(result["answer"])


if __name__ == "__main__":
    main()
