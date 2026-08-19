"""CLI entrypoint: `uv run swarm-agent "<message>"`.

Each invocation uses a fresh thread — to see the swarm's persistent-active-agent behavior (a
follow-up in the *same* thread resuming with whichever specialist last took control), pass the
same thread id across multiple invocations of `build_swarm` in a script rather than via this CLI,
which is one-shot by design like `react_agent`'s and `supervisor_agent`'s.
"""

from __future__ import annotations

import sys
import uuid

from agents_common import configure_mlflow, get_checkpointer
import mlflow

from swarm_agent.graph import (
    AGENTS,
    EXPERIMENT_NAME,
    build_swarm,
    invoke_config,
    link_prompts_to_trace,
    load_agent_prompt_version,
    prompt_text,
)

_MIN_ARGC = 2


def main() -> None:
    """Run one turn of the swarm against a fresh thread and print the active agent's answer."""
    if len(sys.argv) < _MIN_ARGC:
        print('Usage: swarm-agent "<message>"', file=sys.stderr)
        raise SystemExit(1)

    message = sys.argv[1]
    configure_mlflow(EXPERIMENT_NAME)

    # Fetched once, up front, rather than through build_swarm's default lookup: this way the
    # exact PromptVersions used to build both agents are on hand afterwards to link to the trace
    # this invocation produces (see link_prompts_to_trace below). Same convention as
    # supervisor_agent.__main__.
    prompt_versions = {name: load_agent_prompt_version(name) for name in AGENTS}

    with get_checkpointer() as checkpointer:
        swarm = build_swarm(
            checkpointer=checkpointer,
            agent_prompts={name: prompt_text(version) for name, version in prompt_versions.items()},
        )
        config = invoke_config(str(uuid.uuid4()))
        result = swarm.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
        link_prompts_to_trace(prompt_versions, mlflow.get_last_active_trace_id())
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
