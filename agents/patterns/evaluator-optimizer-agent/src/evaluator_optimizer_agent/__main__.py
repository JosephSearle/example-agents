"""CLI entrypoint: `uv run evaluator-optimizer-agent "<task>" "<criteria>"`."""

from __future__ import annotations

import sys
import uuid

from agents_common import configure_mlflow, get_checkpointer
import mlflow

from evaluator_optimizer_agent.graph import (
    EXPERIMENT_NAME,
    STEPS,
    build_evaluator_optimizer_graph,
    invoke_config,
    link_prompts_to_trace,
    load_step_prompt_version,
    prompt_text,
)

_MIN_ARGC = 3


def main() -> None:
    """Run the generate/evaluate loop until approved or the iteration cap is hit, and print it."""
    if len(sys.argv) < _MIN_ARGC:
        print('Usage: evaluator-optimizer-agent "<task>" "<criteria>"', file=sys.stderr)
        raise SystemExit(1)

    task, criteria = sys.argv[1], sys.argv[2]
    configure_mlflow(EXPERIMENT_NAME)

    # Fetched once, up front, rather than through build_evaluator_optimizer_graph's default
    # lookup: this way the exact PromptVersions used to build the graph are on hand afterwards to
    # link to the trace this invocation produces (see link_prompts_to_trace below). Same
    # convention as orchestrator_workers_agent.__main__.
    prompt_versions = {step: load_step_prompt_version(step) for step in STEPS}

    with get_checkpointer() as checkpointer:
        graph = build_evaluator_optimizer_graph(
            checkpointer=checkpointer,
            step_prompts={step: prompt_text(version) for step, version in prompt_versions.items()},
        )
        config = invoke_config(str(uuid.uuid4()))
        result = graph.invoke(  # type: ignore[call-overload]
            {
                "task": task,
                "criteria": criteria,
                "response": "",
                "feedback": "",
                "approved": False,
                "iteration": 0,
            },
            config=config,
        )
        link_prompts_to_trace(prompt_versions, mlflow.get_last_active_trace_id())
        status = "approved" if result["approved"] else "max iterations reached"
        print(f"[{status} after {result['iteration']} iteration(s)]\n{result['response']}")


if __name__ == "__main__":
    main()
