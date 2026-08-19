"""Provision MLflow GenAI eval datasets from the git-tracked JSONL seed files.

Reads packages/mlflow-server/datasets/. Each <agent-name>.jsonl file becomes an MLflow dataset named <agent-name>, associated with the
<agent-name> experiment (created if missing), with its rows merged in via
EvaluationDataset.merge_records(). Unlike provision_gateway_route.py, this uses the mlflow Python
client directly rather than raw REST calls, since mlflow.genai.datasets is a first-class Python
API — no undocumented proto-backed gateway routes to hand-roll here.

Idempotent by design: re-running is safe. merge_records() is a merge/upsert, not an append, so
re-running after editing a JSONL file re-syncs the dataset to match.

Run after `make up` (needs a live mlflow-server):

    uv run python packages/mlflow-server/scripts/provision_datasets.py

Known gap: .github/workflows/eval.yml runs `pytest -m eval` against a real, pre-existing external
MLflow instance (secrets.MLFLOW_TRACKING_URI) and does not run this script — same pre-existing gap
as provision_gateway_route.py / make provision-gateway. Datasets must be provisioned against that
instance separately. Not solved here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents_common.config import get_settings
from agents_common.logging import configure_logging
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import EvaluationDataset, create_dataset, get_dataset
import structlog

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

_logger = structlog.get_logger(__name__)


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _get_or_create_dataset(name: str, experiment_id: str) -> EvaluationDataset:
    # mlflow.genai.datasets' functions are untyped at the stub level (return Any) —
    # same library-boundary gap as react_agent.graph's `# type: ignore[arg-type]`.
    try:
        return get_dataset(name=name)  # type: ignore[no-any-return]
    except MlflowException as e:
        if e.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return create_dataset(  # type: ignore[no-any-return]
            name=name, experiment_id=experiment_id, tags={"source": "seed-jsonl"}
        )


def main() -> None:
    """Sync every dataset JSONL in DATASETS_DIR into MLflow's dataset registry."""
    configure_logging()
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    for jsonl_path in sorted(DATASETS_DIR.glob("*.jsonl")):
        agent_name = jsonl_path.stem
        _logger.info("provisioning_dataset", agent=agent_name, source=jsonl_path.name)
        experiment = mlflow.set_experiment(agent_name)
        dataset = _get_or_create_dataset(agent_name, experiment.experiment_id)
        records = _load_records(jsonl_path)
        dataset.merge_records(records)
        _logger.info(
            "dataset_merged",
            agent=agent_name,
            record_count=len(records),
            experiment_id=experiment.experiment_id,
        )

    _logger.info("done")


if __name__ == "__main__":
    main()
