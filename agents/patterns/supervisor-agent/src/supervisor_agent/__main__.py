"""CLI entrypoint: `uv run supervisor-agent "<message>"`."""

from __future__ import annotations

import sys
import uuid

from agents_common import configure_mlflow, get_checkpointer
import mlflow

from supervisor_agent.graph import (
    AGENTS,
    EXPERIMENT_NAME,
    build_supervisor,
    invoke_config,
    link_prompts_to_trace,
    load_agent_prompt_version,
    prompt_text,
)

_MIN_ARGC = 2


def main() -> None:
    """Run one turn of the supervisor against a fresh thread and print its final answer."""
    if len(sys.argv) < _MIN_ARGC:
        print('Usage: supervisor-agent "<message>"', file=sys.stderr)
        raise SystemExit(1)

    message = sys.argv[1]
    configure_mlflow(EXPERIMENT_NAME)

    # Fetched once, up front, rather than through build_supervisor's default lookup: this way
    # the exact PromptVersions used to build the supervisor and both sub-agents are on hand
    # afterwards to link to the trace this invocation produces (see link_prompts_to_trace
    # below). Same convention as orchestrator_workers_agent.__main__.
    prompt_versions = {name: load_agent_prompt_version(name) for name in AGENTS}

    with get_checkpointer() as checkpointer:
        supervisor = build_supervisor(
            checkpointer=checkpointer,
            agent_prompts={name: prompt_text(version) for name, version in prompt_versions.items()},
        )
        config = invoke_config(str(uuid.uuid4()))
        result = supervisor.invoke(
            {"messages": [{"role": "user", "content": message}]}, config=config
        )
        link_prompts_to_trace(prompt_versions, mlflow.get_last_active_trace_id())
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
