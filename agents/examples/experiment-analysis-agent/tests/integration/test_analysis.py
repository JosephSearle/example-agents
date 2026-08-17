"""Integration test: a full analysis run against a live experiment produces a real report.

Heavier than this repo's other `tests/integration` suites: those need only Postgres
(`.github/workflows/integration.yml` provisions just that), but this test needs a live
`mlflow-mcp` server (spawned as a subprocess, talking to a real MLflow tracking server), this
agent's system prompt provisioned into MLflow's prompt registry (`make provision-prompts`, which
`make up` already runs automatically), at least one real trace in the target experiment, plus a
real call to the MLflow AI Gateway model. Not wired into `integration.yml` for that reason — same
acknowledged gap as `provision_monitors.py`'s docstring notes for CI running against a real
external MLflow instance. Run locally after `make up` + `make provision-gateway` + `make demo`
(to seed at least one react-agent trace):

    uv run pytest -m integration --no-header -k test_analysis
"""

from __future__ import annotations

from experiment_analysis_agent.graph import REPORT_PATH, run_analysis
import pytest

pytestmark = pytest.mark.integration


def test_run_analysis_produces_a_nonempty_report() -> None:
    report = run_analysis("react-agent")

    assert report.strip(), f"expected a non-empty report written to {REPORT_PATH}"
