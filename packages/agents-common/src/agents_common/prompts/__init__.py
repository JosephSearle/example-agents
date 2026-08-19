"""Shared MLflow prompt-registry helpers.

Every agent in this repo registers its prompt(s) under `prompts:/<name>@<alias>` (see
packages/mlflow-server/scripts/provision_prompts.py) and needs the same three operations around
that: load a `PromptVersion` by registry name, narrow it to plain text, and link the version(s)
used by an invocation to that invocation's trace. Originally implemented once per agent
(`react_agent.graph` and `prompt_chaining_agent.graph`); pulled up here once a second agent
duplicated it verbatim, per the Rule of Three trade-off of not extracting on the first repeat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow
from mlflow import MlflowClient
from mlflow.genai.prompts import load_prompt
import structlog

from agents_common.config import get_settings

_logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from mlflow.entities.model_registry import PromptVersion

PRODUCTION_ALIAS = "production"


def load_prompt_version(
    registry_name: str, *, experiment_name: str, alias: str = PRODUCTION_ALIAS
) -> PromptVersion:
    """Fetch a prompt version from the MLflow prompt registry.

    Sets the tracking URI and active experiment itself rather than assuming
    `agents_common.observability.configure_mlflow` already ran, so it works whether the caller is
    a `__main__` entrypoint (which does call it first) or a test that only sets up a
    checkpointer — and so the loaded prompt version is linked to `experiment_name` (MLflow only
    links a prompt to whichever experiment is *active* at load time, not implicitly by name
    match).

    Returns the full `PromptVersion` (not just its text) so a caller running the agent can pass
    it to `link_prompt_to_trace` afterwards.

    Args:
        registry_name: The prompt's registered name, e.g. `"react-agent"` or
            `"prompt-chaining-agent-outline"`.
        experiment_name: The MLflow experiment to make active before loading, so the fetch is
            attributed correctly.
        alias: Prompt registry alias to load. Defaults to the production alias.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    return load_prompt(f"prompts:/{registry_name}@{alias}")  # type: ignore[no-any-return]


def prompt_text(prompt_version: PromptVersion) -> str:
    """Narrow a `PromptVersion`'s template to plain text.

    `PromptVersion.template` is typed as `str | list[dict]` because MLflow's prompt registry also
    supports chat-messages-list templates; every agent in this repo registers plain-text prompts,
    so a non-`str` template indicates a registry/provisioning mismatch worth failing loudly on
    rather than silently mishandling.
    """
    template = prompt_version.template
    if not isinstance(template, str):
        msg = f"Expected a plain-text prompt template for {prompt_version.name!r}, got {type(template).__name__}"
        raise TypeError(msg)
    return template


def link_prompts_to_trace(prompt_versions: list[PromptVersion], trace_id: str | None) -> None:
    """Link prompt version(s) to a trace so the MLflow UI's trace view shows them under "Prompts".

    `trace_id` is typically `mlflow.get_last_active_trace_id()`, called right after
    `.invoke(...)` returns (autologging via `mlflow.langchain.autolog()` — see
    `agents_common.observability.configure_mlflow` — creates one trace per invocation). A `None`
    trace_id (autologging disabled, or nothing traced yet) is a no-op rather than an error, since
    linking is an enhancement to an already-successful invocation, not something that invocation
    should fail over.

    For the same reason, any error the MLflow backend call raises is logged and swallowed rather
    than propagated — most commonly a `RESOURCE_DOES_NOT_EXIST` `RestException`, since trace
    export runs on `mlflow.tracing.export.async_export_queue`'s background thread and can still be
    in flight when this is called right after `.invoke()` returns. Every `__main__.py` calls this
    *before* printing the invocation's actual result, so letting that race crash the process would
    throw away a result that already computed successfully.
    """
    if trace_id is None:
        return
    try:
        MlflowClient().link_prompt_versions_to_trace(
            prompt_versions=prompt_versions, trace_id=trace_id
        )
    except Exception:
        _logger.warning("link_prompt_versions_to_trace_failed", trace_id=trace_id, exc_info=True)
