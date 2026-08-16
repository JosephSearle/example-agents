"""CLI entrypoint: `uv run prompt-chaining-agent "<topic>"`."""

from __future__ import annotations

import sys
import uuid

from agents_common import configure_mlflow, get_checkpointer
import mlflow

from prompt_chaining_agent.graph import (
    EXPERIMENT_NAME,
    STEPS,
    build_chain,
    invoke_config,
    link_prompts_to_trace,
    load_step_prompt_version,
    prompt_text,
)

_MIN_ARGC = 2


def main() -> None:
    """Run the chain once against a fresh thread and print the final polished text."""
    if len(sys.argv) < _MIN_ARGC:
        print('Usage: prompt-chaining-agent "<topic>"', file=sys.stderr)
        raise SystemExit(1)

    topic = sys.argv[1]
    configure_mlflow(EXPERIMENT_NAME)

    # Fetched once, up front, rather than through build_chain's default lookup: this way the
    # exact PromptVersions used to build the chain are on hand afterwards to link to the trace
    # this invocation produces (see link_prompts_to_trace below).
    prompt_versions = {step: load_step_prompt_version(step) for step in STEPS}

    with get_checkpointer() as checkpointer:
        chain = build_chain(
            checkpointer=checkpointer,
            step_prompts={step: prompt_text(version) for step, version in prompt_versions.items()},
        )
        config = invoke_config(str(uuid.uuid4()))
        result = chain.invoke(
            {"topic": topic, "outline": "", "draft": "", "final": ""}, config=config
        )  # type: ignore[call-overload]
        link_prompts_to_trace(prompt_versions, mlflow.get_last_active_trace_id())
        print(result["final"])


if __name__ == "__main__":
    main()
