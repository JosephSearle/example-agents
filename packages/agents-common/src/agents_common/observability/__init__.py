"""MLflow wiring shared by every agent: tracking URI, per-agent experiment, and autologging."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow
from mlflow.genai.scorers import ScorerSamplingConfig

from agents_common.config import Settings, get_settings

if TYPE_CHECKING:
    from mlflow.genai.scorers import Scorer


def configure_mlflow(experiment_name: str, *, settings: Settings | None = None) -> None:
    """Point MLflow at the tracking server and enable LangChain/LangGraph autologging.

    Call once at process startup (agent CLI entrypoint, API server startup, or the top of a
    test session). After this, every `create_agent`/graph invocation is traced automatically —
    no manual span instrumentation required in agent code.

    Each pattern gets its own MLflow experiment rather than sharing one repo-wide experiment —
    `react-agent`, `supervisor-agent`, etc. are meaningfully different systems, and a shared
    experiment would mix their runs/metrics together in one undifferentiated stream. Define the
    name as a constant in the calling agent's package (see `react_agent.EXPERIMENT_NAME`)
    rather than sourcing it from `.env` — it identifies which agent produced a run, not
    something a dev should need to configure per environment.

    Args:
        experiment_name: The MLflow experiment this agent's runs belong to, e.g. "react-agent".
        settings: Override settings; defaults to `get_settings()`.
    """
    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.langchain.autolog()


def register_production_monitors(
    experiment_name: str,
    scorers: list[tuple[Scorer, float]],
    *,
    settings: Settings | None = None,
) -> None:
    """Register and start scorers that continuously score live production traces.

    This is the "monitor" half of https://mlflow.org/docs/latest/genai/eval-monitor/ — distinct
    from `mlflow.genai.evaluate()` in each pattern's `tests/evals/test_quality.py`, which scores
    a fixed dataset in CI. This instead runs a scorer against a sampled slice of real traffic as
    it's traced through `configure_mlflow`'s `mlflow.langchain.autolog()`, so quality is trended
    on production usage rather than only measured against a static eval set.

    Call once per agent (e.g. from `provision_monitors.py`), not from the agent's own runtime
    startup path — starting/stopping monitors is a deploy-time configuration change, not
    something that should happen on every process boot.

    Idempotent by design: re-running is safe. `Scorer.register()` versions by name — calling it
    again for a scorer that's already registered creates a new version (picking up any edit to
    its guideline text, model, etc.) rather than erroring, and `.start()` re-applies the sampling
    config to whichever version is now current. So editing a `PRODUCTION_SCORERS` entry and
    re-running this always converges the server to match the code, without a separate
    get-then-branch step.

    Args:
        experiment_name: The MLflow experiment to attach these monitors to, e.g. "react-agent"
            (same constant as `configure_mlflow`'s `experiment_name`).
        scorers: `(scorer, sample_rate)` pairs — `sample_rate` is the fraction of production
            traces (0.0-1.0) this scorer runs against. Judge-backed scorers (`Guidelines`,
            `Safety`, ...) must have `model=` in `gateway:/<route>` form pointing at the agent's
            own provisioned AI Gateway route — `Scorer.start()` rejects any other model form
            (confirmed against a live mlflow-server), unlike `mlflow.genai.evaluate()`'s judges
            in each pattern's `tests/evals/test_quality.py`, which use `openai:/<route>` via the
            `OPENAI_API_BASE` env var instead.
        settings: Override settings; defaults to `get_settings()`.
    """
    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)

    for scorer, sample_rate in scorers:
        registered = scorer.register(experiment_id=experiment.experiment_id)
        registered.start(sampling_config=ScorerSamplingConfig(sample_rate=sample_rate))
        print(f"  registered + started monitor '{scorer.name}' (sample_rate={sample_rate})")
