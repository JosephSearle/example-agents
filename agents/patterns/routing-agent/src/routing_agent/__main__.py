"""CLI entrypoint: `uv run routing-agent "<support ticket message>"`."""

from __future__ import annotations

import sys
import uuid

from agents_common import configure_mlflow, get_checkpointer
import mlflow

from routing_agent.graph import (
    CATEGORIES,
    EXPERIMENT_NAME,
    build_router,
    invoke_config,
    link_prompts_to_trace,
    load_route_prompt_version,
    prompt_text,
)

_MIN_ARGC = 2


def main() -> None:
    """Classify a ticket, dispatch it to its handler, and print the response."""
    if len(sys.argv) < _MIN_ARGC:
        print('Usage: routing-agent "<support ticket message>"', file=sys.stderr)
        raise SystemExit(1)

    message = sys.argv[1]
    configure_mlflow(EXPERIMENT_NAME)

    # Fetched once, up front, rather than through build_router's default lookup: this way the
    # exact PromptVersions used to build the graph are on hand afterwards to link to the trace
    # this invocation produces (see link_prompts_to_trace below). Only the category that actually
    # fires needs linking, but fetching all three up front keeps this symmetric with
    # prompt_chaining_agent's __main__ and avoids a second registry round-trip after routing.
    prompt_versions = {category: load_route_prompt_version(category) for category in CATEGORIES}

    with get_checkpointer() as checkpointer:
        router = build_router(
            checkpointer=checkpointer,
            route_prompts={
                category: prompt_text(version) for category, version in prompt_versions.items()
            },
        )
        config = invoke_config(str(uuid.uuid4()))
        result = router.invoke({"message": message, "category": "", "response": ""}, config=config)  # type: ignore[call-overload]
        link_prompts_to_trace(
            {result["category"]: prompt_versions[result["category"]]},
            mlflow.get_last_active_trace_id(),
        )
        print(f"[{result['category']}] {result['response']}")


if __name__ == "__main__":
    main()
