"""CLI entrypoint: `uv run experiment-analysis-agent <experiment-name>`."""

from __future__ import annotations

from pathlib import Path
import sys

from agents_common import configure_mlflow

from experiment_analysis_agent.graph import (
    EXPERIMENT_NAME,
    load_system_prompt_version,
    run_analysis,
)

_MIN_ARGC = 2


def main() -> None:
    """Analyze one MLflow experiment's traces and write the resulting report to disk."""
    if len(sys.argv) < _MIN_ARGC:
        print("Usage: experiment-analysis-agent <experiment-name>", file=sys.stderr)
        raise SystemExit(1)

    target_experiment = sys.argv[1]
    configure_mlflow(EXPERIMENT_NAME)

    # Fetched once, up front, rather than through build_agent's default lookup: this way the
    # exact PromptVersion used to build the agent is on hand afterwards to link to the trace
    # this invocation produces — same pattern as react_agent.__main__.
    prompt_version = load_system_prompt_version()
    report = run_analysis(target_experiment, prompt_version=prompt_version)

    report_path = Path(f"report-{target_experiment}.md")
    report_path.write_text(report)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
