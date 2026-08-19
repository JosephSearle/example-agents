"""Provision an MLflow AI Gateway route backed by a self-hosted OpenAI-compatible model.

The gateway is baked into the tracking server (mlflow[genai]>=3.1, mounted at
/gateway/mlflow/v1) but nothing provisions a route automatically — that's a REST-driven,
one-time setup step, normally done through the MLflow web UI. This script does the same four
calls from the CLI so it's repeatable and scriptable in dev:

    1. POST .../gateway/secrets/create           — store the upstream model's API key
    2. POST .../gateway/model-definitions/create  — point a named model at that secret
    3. POST .../gateway/endpoints/create           — create the route, wired to that model

Run once against a running mlflow-server (`docker compose up -d mlflow` first):

    uv run python packages/mlflow-server/scripts/provision_gateway_route.py

All inputs come from the environment (see .env.example): MLFLOW_TRACKING_URI,
MLFLOW_TRACKING_TOKEN (only if the server has auth enabled), SELFHOSTED_MODEL_BASE_URL,
SELFHOSTED_MODEL_API_KEY, SELFHOSTED_MODEL_NAME, and GATEWAY_ROUTE_NAME.

Not idempotent by design — MLflow has no upsert-by-name for these resources. Re-running
after a route already exists will fail on the endpoint-create call (name collision); delete
it first via the UI or the /gateway/endpoints/delete API if you need to re-provision.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from agents_common.logging import configure_logging
from dotenv import load_dotenv
import requests
import structlog

# Plain `os.environ` reads, unlike `agents_common.config.Settings` (pydantic-settings' own
# env_file loading), so .env has to be loaded explicitly here for `uv run python
# packages/mlflow-server/scripts/provision_gateway_route.py` to pick up repo-root .env values.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

_logger = structlog.get_logger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_TRACKING_TOKEN = os.environ.get("MLFLOW_TRACKING_TOKEN", "")

SELFHOSTED_MODEL_BASE_URL = os.environ["SELFHOSTED_MODEL_BASE_URL"]
SELFHOSTED_MODEL_API_KEY = os.environ.get("SELFHOSTED_MODEL_API_KEY", "unused")
SELFHOSTED_MODEL_NAME = os.environ.get("SELFHOSTED_MODEL_NAME", "gpt-oss-120b")
GATEWAY_ROUTE_NAME = os.environ.get("GATEWAY_ROUTE_NAME", SELFHOSTED_MODEL_NAME)

_session = requests.Session()
if MLFLOW_TRACKING_TOKEN:
    _session.headers["Authorization"] = f"Bearer {MLFLOW_TRACKING_TOKEN}"


_HTTP_BAD_REQUEST = 400


def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    url = f"{MLFLOW_TRACKING_URI}/api/3.0/mlflow/gateway/{path}"
    resp = _session.post(url, json=body, timeout=30)
    if resp.status_code == _HTTP_BAD_REQUEST and "RESOURCE_ALREADY_EXISTS" in resp.text:
        _logger.error(
            "gateway_route_already_provisioned",
            route=GATEWAY_ROUTE_NAME,
            path=path,
            hint=(
                "This script isn't idempotent — delete the existing secret/model-definition/"
                "endpoint via the MLflow UI first if you need to re-provision, or leave it "
                "as-is if it's already working."
            ),
        )
        sys.exit(1)
    if not resp.ok:
        _logger.error("gateway_post_failed", url=url, status_code=resp.status_code, body=resp.text)
        resp.raise_for_status()
    result: dict[str, Any] = resp.json()
    return result


def main() -> None:
    """Create the gateway secret, model definition, and endpoint, in that order."""
    configure_logging()
    _logger.info("creating_gateway_secret", route=GATEWAY_ROUTE_NAME)
    secret = _post(
        "secrets/create",
        {
            "secret_name": f"{GATEWAY_ROUTE_NAME}-key",
            # {"api_key": ...} is the shape mlflow's "openai" provider reads for auth —
            # see mlflow.gateway.config._AuthConfigKey.API_KEY.
            "secret_value": {"api_key": SELFHOSTED_MODEL_API_KEY},
            "provider": "openai",
            # {"api_base": ...} tells the "openai" provider to call our self-hosted endpoint
            # instead of api.openai.com — see mlflow.server.gateway_api._build_endpoint_config.
            "auth_config": {"api_base": SELFHOSTED_MODEL_BASE_URL},
        },
    )
    secret_id = secret["secret"]["secret_id"]
    _logger.info("gateway_secret_created", secret_id=secret_id)

    _logger.info("creating_gateway_model_definition", route=GATEWAY_ROUTE_NAME)
    model_definition = _post(
        "model-definitions/create",
        {
            "name": GATEWAY_ROUTE_NAME,
            "secret_id": secret_id,
            "provider": "openai",
            "model_name": SELFHOSTED_MODEL_NAME,
        },
    )
    model_definition_id = model_definition["model_definition"]["model_definition_id"]
    _logger.info("gateway_model_definition_created", model_definition_id=model_definition_id)

    _logger.info("creating_gateway_endpoint", route=GATEWAY_ROUTE_NAME)
    endpoint = _post(
        "endpoints/create",
        {
            "name": GATEWAY_ROUTE_NAME,
            "model_configs": [
                {
                    "model_definition_id": model_definition_id,
                    "linkage_type": "PRIMARY",
                    "weight": 1.0,
                }
            ],
        },
    )
    _logger.info("gateway_endpoint_created", endpoint_id=endpoint["endpoint"]["endpoint_id"])

    _logger.info(
        "done",
        route=GATEWAY_ROUTE_NAME,
        chat_completions_url=f"{MLFLOW_TRACKING_URI}/gateway/mlflow/v1/chat/completions",
    )


if __name__ == "__main__":
    main()
