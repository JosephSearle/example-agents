"""Register and start production quality monitors for every agent that defines them.

This is the "monitor" half of https://mlflow.org/docs/latest/genai/eval-monitor/, complementing
`tests/evals/test_quality.py`'s offline `mlflow.genai.evaluate()` runs (a fixed dataset, scored
in CI). Monitors instead score a sampled slice of *live* production traces continuously, via
`Scorer.register()` + `.start()` — see `agents_common.observability.register_production_monitors`.

Each agent declares its own `PRODUCTION_SCORERS` constant in its `graph.py`, next to
`EXPERIMENT_NAME`/`GATEWAY_ROUTE` — a list of `(scorer, sample_rate)` pairs. Unlike
provision_prompts.py / provision_datasets.py (which are driven off files in this directory, with
no import dependency on any agent package), this script imports each agent package directly to
read that constant — all three are uv workspace members already on this venv's path, the same
way `tests/evals/test_quality.py` imports `EXPERIMENT_NAME`/`GATEWAY_ROUTE` from `graph.py`.

Idempotent by design: re-running is safe (see `register_production_monitors`'s docstring) — a
scorer already registered under the same name has its sampling config updated in place rather
than being re-registered.

Not wired into `make up` (unlike provision-prompts): starting/stopping production monitors is a
deliberate "turn this feature on" action, not an idempotent sync a dev wants running on every
`make up`. Run manually after `make up` + `make provision-gateway`:

    uv run python packages/mlflow-server/scripts/provision_monitors.py

Known gap: .github/workflows/eval.yml runs `pytest -m eval` against a real, pre-existing external
MLflow instance (secrets.MLFLOW_TRACKING_URI) and does not run this script — same pre-existing gap
as provision_gateway_route.py / provision_datasets.py / provision_prompts.py. Monitors must be
provisioned against that instance separately. Not solved here.
"""

from __future__ import annotations

from agents_common import register_production_monitors

_AGENTS = ["react_agent", "routing_agent", "prompt_chaining_agent"]


def main() -> None:
    """Register + start `PRODUCTION_SCORERS` for every agent package that defines them."""
    for module_name in _AGENTS:
        module = __import__(f"{module_name}.graph", fromlist=["graph"])
        experiment_name = module.EXPERIMENT_NAME
        scorers = module.PRODUCTION_SCORERS

        print(f"Provisioning {len(scorers)} monitor(s) for '{experiment_name}'...")
        register_production_monitors(experiment_name, scorers)

    print("\nDone.")


if __name__ == "__main__":
    main()
