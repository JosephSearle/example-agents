"""CLI entrypoint: `uv run parallelization-agent "<incident text>"`, or `--voting "<prompt>"`."""

from __future__ import annotations

import sys
import uuid

from agents_common import configure_mlflow, get_checkpointer
import mlflow

from parallelization_agent.graph import (
    EXPERIMENT_NAME,
    SECTIONS,
    build_sectioning_graph,
    build_voting_graph,
    invoke_config,
    link_prompts_to_trace,
    load_section_prompt_version,
    prompt_text,
)

_MIN_ARGC = 2
_USAGE = (
    'Usage: parallelization-agent "<incident text>" | parallelization-agent --voting "<prompt>"'
)


def main() -> None:
    """Run the sectioning graph by default, or the voting graph with `--voting`."""
    if len(sys.argv) < _MIN_ARGC:
        print(_USAGE, file=sys.stderr)
        raise SystemExit(1)

    configure_mlflow(EXPERIMENT_NAME)

    if sys.argv[1] == "--voting":
        if len(sys.argv) < _MIN_ARGC + 1:
            print(_USAGE, file=sys.stderr)
            raise SystemExit(1)
        prompt = sys.argv[2]
        with get_checkpointer() as checkpointer:
            voting = build_voting_graph(checkpointer=checkpointer)
            config = invoke_config(str(uuid.uuid4()))
            result = voting.invoke({"prompt": prompt, "attempts": [], "verdict": ""}, config=config)  # type: ignore[call-overload]
            print(result["verdict"])
        return

    incident_text = sys.argv[1]

    # Fetched once, up front, rather than through build_sectioning_graph's default lookup: this
    # way the exact PromptVersions used to build the graph are on hand afterwards to link to the
    # trace this invocation produces (see link_prompts_to_trace below). All three sections always
    # fire (unlike routing-agent, where only one handler fires per invocation), so all three
    # versions get linked.
    prompt_versions = {section: load_section_prompt_version(section) for section in SECTIONS}

    with get_checkpointer() as checkpointer:
        sectioning = build_sectioning_graph(
            checkpointer=checkpointer,
            section_prompts={
                section: prompt_text(version) for section, version in prompt_versions.items()
            },
        )
        config = invoke_config(str(uuid.uuid4()))
        result = sectioning.invoke(  # type: ignore[call-overload]
            {
                "incident_text": incident_text,
                "summary": "",
                "severity": "",
                "action_items": [],
                "report": "",
            },
            config=config,
        )
        link_prompts_to_trace(prompt_versions, mlflow.get_last_active_trace_id())
        print(result["report"])


if __name__ == "__main__":
    main()
