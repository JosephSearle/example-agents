"""Provision MLflow prompt-registry entries from the git-tracked prompt text files.

Reads packages/mlflow-server/prompts/. Each <agent-name>.txt file is registered in MLflow's
prompt registry under the name <agent-name>, with a `production` alias pointed at the version
matching that file's current content, and linked to the <agent-name> experiment (created if
missing) so its versions show up under that experiment's Prompts tab in the MLflow UI — MLflow
only links a prompt to whichever experiment is *active* (`mlflow.set_experiment`) at the moment
`register_prompt`/`load_prompt` is called, it's not implicit from the prompt name matching an
experiment name. Agents load their system prompt at runtime via
`mlflow.genai.load_prompt("prompts:/<agent-name>@production")` rather than importing it from a
Python module — see agents/patterns/react-agent/src/react_agent/graph.py's `load_system_prompt`.

Agents with more than one prompt (e.g. a chain with a distinct prompt per step) use one
subdirectory per agent instead: packages/mlflow-server/prompts/<agent-name>/<step-name>.txt is
registered as prompt name <agent-name>-<step-name>, still linked to the single <agent-name>
experiment (set once per subdirectory, so all of that agent's step prompts land on the same
experiment's Prompts tab) — see
agents/patterns/prompt-chaining-agent/src/prompt_chaining_agent/graph.py's `load_step_prompt`.

Idempotent by design: re-running is safe. Before registering, this compares the file's content
against the template of the current `production`-aliased version (if any) and skips registering
a new version when they already match, so re-running `make up` doesn't create a new prompt
version (or move the alias) on every run — only an actual edit to a prompt file does.

Run after `make up` (needs a live mlflow-server); wired into `make up` itself via the
`provision-prompts` target, so this normally runs automatically:

    uv run python packages/mlflow-server/scripts/provision_prompts.py

Known gap: .github/workflows/eval.yml runs `pytest -m eval` against a real, pre-existing external
MLflow instance (secrets.MLFLOW_TRACKING_URI) and does not run this script — same pre-existing gap
as provision_gateway_route.py / provision_datasets.py. Prompts must be provisioned against that
instance separately. Not solved here.
"""

from __future__ import annotations

from pathlib import Path

from agents_common.config import get_settings
from agents_common.logging import configure_logging
import mlflow
from mlflow.genai.prompts import load_prompt, register_prompt, set_prompt_alias
import structlog

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_PRODUCTION_ALIAS = "production"

_logger = structlog.get_logger(__name__)


def _current_production_template(name: str) -> str | None:
    prompt = load_prompt(f"prompts:/{name}@{_PRODUCTION_ALIAS}", allow_missing=True)
    return None if prompt is None else prompt.template


def _provision_one(name: str, text: str, *, source: str) -> None:
    if _current_production_template(name) == text:
        _logger.info("prompt_unchanged", name=name)
        return

    _logger.info("provisioning_prompt", name=name, source=source)
    version = register_prompt(
        name=name,
        template=text,
        commit_message=f"Provisioned from {source}",
        tags={"source": "seed-file"},
    )
    set_prompt_alias(name, _PRODUCTION_ALIAS, version.version)
    _logger.info("prompt_registered", name=name, version=version.version, alias=_PRODUCTION_ALIAS)


def main() -> None:
    """Sync every prompt text file in PROMPTS_DIR into MLflow's prompt registry."""
    configure_logging()
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    for prompt_path in sorted(PROMPTS_DIR.glob("*.txt")):
        name = prompt_path.stem

        # Must be set before the register_prompt/load_prompt calls below, since MLflow links a
        # prompt version to whichever experiment is active at call time.
        mlflow.set_experiment(name)
        _provision_one(name, prompt_path.read_text(), source=f"prompts/{prompt_path.name}")

    for agent_dir in sorted(p for p in PROMPTS_DIR.iterdir() if p.is_dir()):
        # One experiment per agent, set once so every step prompt in this subdirectory links to
        # the same experiment's Prompts tab rather than one experiment per step.
        mlflow.set_experiment(agent_dir.name)

        for prompt_path in sorted(agent_dir.glob("*.txt")):
            name = f"{agent_dir.name}-{prompt_path.stem}"
            _provision_one(
                name,
                prompt_path.read_text(),
                source=f"prompts/{agent_dir.name}/{prompt_path.name}",
            )

    _logger.info("done")


if __name__ == "__main__":
    main()
