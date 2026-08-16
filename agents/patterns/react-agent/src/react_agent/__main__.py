"""CLI entrypoint: `uv run react-agent "<message>"`."""

from __future__ import annotations

import sys
import uuid

from agents_common import configure_mlflow, get_checkpointer
import mlflow

from react_agent.graph import (
    EXPERIMENT_NAME,
    build_agent,
    extract_response,
    invoke_config,
    link_prompt_to_trace,
    load_system_prompt_version,
    prompt_text,
)

_MIN_ARGC = 2


def main() -> None:
    """Run one turn of the agent against a fresh thread and print the structured response."""
    if len(sys.argv) < _MIN_ARGC:
        print('Usage: react-agent "<message>"', file=sys.stderr)
        raise SystemExit(1)

    message = sys.argv[1]
    configure_mlflow(EXPERIMENT_NAME)

    # Fetched once, up front, rather than through build_agent's default lookup: this way the
    # exact PromptVersion used to build the agent is on hand afterwards to link to the trace
    # this invocation produces (see link_prompt_to_trace below).
    prompt_version = load_system_prompt_version()

    with get_checkpointer() as checkpointer:
        agent = build_agent(checkpointer=checkpointer, system_prompt=prompt_text(prompt_version))
        config = invoke_config(str(uuid.uuid4()))
        result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
        link_prompt_to_trace(prompt_version, mlflow.get_last_active_trace_id())
        print(extract_response(result))


if __name__ == "__main__":
    main()
