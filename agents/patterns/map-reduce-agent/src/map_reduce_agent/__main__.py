"""CLI entrypoint: `uv run map-reduce-agent "<topic 1>" "<topic 2>" ...`."""

from __future__ import annotations

import sys
import uuid

from agents_common import configure_mlflow, get_checkpointer
import mlflow

from map_reduce_agent.graph import (
    EXPERIMENT_NAME,
    build_map_reduce_graph,
    invoke_config,
    link_prompt_to_trace,
    load_joke_prompt_version,
    prompt_text,
)

_MIN_ARGC = 2


def main() -> None:
    """Generate one joke per topic argument (runtime-determined count) and print the summary."""
    if len(sys.argv) < _MIN_ARGC:
        print('Usage: map-reduce-agent "<topic 1>" "<topic 2>" ...', file=sys.stderr)
        raise SystemExit(1)

    topics = sys.argv[1:]
    configure_mlflow(EXPERIMENT_NAME)

    prompt_version = load_joke_prompt_version()

    with get_checkpointer() as checkpointer:
        graph = build_map_reduce_graph(
            checkpointer=checkpointer, joke_prompt=prompt_text(prompt_version)
        )
        config = invoke_config(str(uuid.uuid4()))
        result = graph.invoke({"topics": topics, "jokes": [], "summary": ""}, config=config)  # type: ignore[call-overload]
        link_prompt_to_trace(prompt_version, mlflow.get_last_active_trace_id())
        print(result["summary"])


if __name__ == "__main__":
    main()
