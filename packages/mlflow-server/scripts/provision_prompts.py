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
import mlflow
from mlflow.genai.prompts import load_prompt, register_prompt, set_prompt_alias

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_PRODUCTION_ALIAS = "production"


def _current_production_template(name: str) -> str | None:
    prompt = load_prompt(f"prompts:/{name}@{_PRODUCTION_ALIAS}", allow_missing=True)
    return None if prompt is None else prompt.template


def main() -> None:
    """Sync every prompt text file in PROMPTS_DIR into MLflow's prompt registry."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    for prompt_path in sorted(PROMPTS_DIR.glob("*.txt")):
        name = prompt_path.stem
        text = prompt_path.read_text()

        # Must be set before the register_prompt/load_prompt calls below, since MLflow links a
        # prompt version to whichever experiment is active at call time.
        mlflow.set_experiment(name)

        if _current_production_template(name) == text:
            print(f"Prompt '{name}' unchanged, skipping.")
            continue

        print(f"Provisioning prompt '{name}' from {prompt_path.name}...")
        version = register_prompt(
            name=name,
            template=text,
            commit_message=f"Provisioned from prompts/{prompt_path.name}",
            tags={"source": "seed-file"},
        )
        set_prompt_alias(name, _PRODUCTION_ALIAS, version.version)
        print(f"  registered version {version.version}, aliased '{_PRODUCTION_ALIAS}' -> it")

    print("\nDone.")


if __name__ == "__main__":
    main()
