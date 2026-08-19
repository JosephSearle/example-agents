"""CLI entrypoint: `uv run orchestrator-workers-agent "<task>"`."""

from __future__ import annotations

import sys
import uuid

from agents_common import configure_mlflow, get_checkpointer
import mlflow

from orchestrator_workers_agent.graph import (
    EXPERIMENT_NAME,
    STEPS,
    build_orchestrator_graph,
    invoke_config,
    link_prompts_to_trace,
    load_step_prompt_version,
    prompt_text,
)

_MIN_ARGC = 2


def main() -> None:
    """Decompose a task, run one worker per orchestrator-decided subtask, and print the synthesis."""
    if len(sys.argv) < _MIN_ARGC:
        print('Usage: orchestrator-workers-agent "<task>"', file=sys.stderr)
        raise SystemExit(1)

    task = sys.argv[1]
    configure_mlflow(EXPERIMENT_NAME)

    # Fetched once, up front, rather than through build_orchestrator_graph's default lookup: this
    # way the exact PromptVersions used to build the graph are on hand afterwards to link to the
    # trace this invocation produces (see link_prompts_to_trace below). Same convention as
    # prompt_chaining_agent.__main__.
    prompt_versions = {step: load_step_prompt_version(step) for step in STEPS}

    with get_checkpointer() as checkpointer:
        graph = build_orchestrator_graph(
            checkpointer=checkpointer,
            step_prompts={step: prompt_text(version) for step, version in prompt_versions.items()},
        )
        config = invoke_config(str(uuid.uuid4()))
        result = graph.invoke(  # type: ignore[call-overload]
            {
                "task": task,
                "analysis": "",
                "subtasks": [],
                "worker_results": [],
                "synthesis": "",
            },
            config=config,
        )
        link_prompts_to_trace(prompt_versions, mlflow.get_last_active_trace_id())
        print(result["synthesis"])


if __name__ == "__main__":
    main()
