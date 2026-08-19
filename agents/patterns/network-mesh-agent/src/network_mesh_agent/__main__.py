"""CLI entrypoint: `uv run network-mesh-agent "<task>"`."""

from __future__ import annotations

import sys
import uuid

from agents_common import configure_mlflow, get_checkpointer
import mlflow

from network_mesh_agent.graph import (
    AGENTS,
    EXPERIMENT_NAME,
    build_mesh_graph,
    invoke_config,
    link_prompts_to_trace,
    load_agent_prompt_version,
    prompt_text,
)

_MIN_ARGC = 2


def main() -> None:
    """Run the mesh on a task and print the writer's final answer."""
    if len(sys.argv) < _MIN_ARGC:
        print('Usage: network-mesh-agent "<task>"', file=sys.stderr)
        raise SystemExit(1)

    task = sys.argv[1]
    configure_mlflow(EXPERIMENT_NAME)

    prompt_versions = {name: load_agent_prompt_version(name) for name in AGENTS}

    with get_checkpointer() as checkpointer:
        graph = build_mesh_graph(
            checkpointer=checkpointer,
            agent_prompts={name: prompt_text(version) for name, version in prompt_versions.items()},
        )
        config = invoke_config(str(uuid.uuid4()))
        result = graph.invoke(  # type: ignore[call-overload]
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
        link_prompts_to_trace(prompt_versions, mlflow.get_last_active_trace_id())
        print(result["final_answer"])


if __name__ == "__main__":
    main()
