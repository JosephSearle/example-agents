"""MLflow wiring shared by every agent: tracking URI, per-agent experiment, and autologging."""

from __future__ import annotations

import mlflow

from agents_common.config import Settings, get_settings


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
