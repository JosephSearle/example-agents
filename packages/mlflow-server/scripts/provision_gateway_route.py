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

from dotenv import load_dotenv
import requests

# Plain `os.environ` reads, unlike `agents_common.config.Settings` (pydantic-settings' own
# env_file loading), so .env has to be loaded explicitly here for `uv run python
# packages/mlflow-server/scripts/provision_gateway_route.py` to pick up repo-root .env values.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

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


def _post(path: str, body: dict) -> dict:
    url = f"{MLFLOW_TRACKING_URI}/api/3.0/mlflow/gateway/{path}"
    resp = _session.post(url, json=body, timeout=30)
    if resp.status_code == _HTTP_BAD_REQUEST and "RESOURCE_ALREADY_EXISTS" in resp.text:
        print(
            f"\n'{GATEWAY_ROUTE_NAME}' is already provisioned (POST {path} returned "
            "RESOURCE_ALREADY_EXISTS). This script isn't idempotent — delete the existing "
            "secret/model-definition/endpoint via the MLflow UI first if you need to "
            "re-provision, or leave it as-is if it's already working.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not resp.ok:
        print(f"POST {url} failed ({resp.status_code}): {resp.text}", file=sys.stderr)
        resp.raise_for_status()
    return resp.json()


def main() -> None:
    """Create the gateway secret, model definition, and endpoint, in that order."""
    print(f"Creating gateway secret for '{GATEWAY_ROUTE_NAME}'...")
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
    print(f"  secret_id={secret_id}")

    print(f"Creating gateway model definition '{GATEWAY_ROUTE_NAME}'...")
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
    print(f"  model_definition_id={model_definition_id}")

    print(f"Creating gateway endpoint '{GATEWAY_ROUTE_NAME}'...")
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
    print(f"  endpoint_id={endpoint['endpoint']['endpoint_id']}")

    print(
        f"\nDone. {GATEWAY_ROUTE_NAME} is now reachable at "
        f"{MLFLOW_TRACKING_URI}/gateway/mlflow/v1/chat/completions (model={GATEWAY_ROUTE_NAME})."
    )


if __name__ == "__main__":
    main()
