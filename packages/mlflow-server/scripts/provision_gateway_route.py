"""Provision MLflow AI Gateway route(s) backed by self-hosted OpenAI-compatible model(s).

The gateway is baked into the tracking server (mlflow[genai]>=3.1, mounted at
/gateway/mlflow/v1) but nothing provisions a route automatically — that's a REST-driven,
one-time setup step, normally done through the MLflow web UI. This script does the same three
calls from the CLI so it's repeatable and scriptable in dev:

    1. POST .../gateway/secrets/create           — store the upstream model's API key
    2. POST .../gateway/model-definitions/create  — point a named model at that secret
    3. POST .../gateway/endpoints/create           — create the route, wired to that model

Run once against a running mlflow-server (`docker compose up -d mlflow` first):

    uv run python packages/mlflow-server/scripts/provision_gateway_route.py

Provisions two routes, both via `_provision_route`: the chat route every agent's
`get_chat_model` calls (`SELFHOSTED_MODEL_*`/`GATEWAY_ROUTE_NAME`, required), and an embeddings
route `agents_common.models.get_embeddings` calls (`SELFHOSTED_EMBEDDING_MODEL_*`/
`EMBEDDING_GATEWAY_ROUTE_NAME`, optional — skipped with a warning if
`SELFHOSTED_EMBEDDING_MODEL_BASE_URL` isn't set, so existing chat-only `.env` setups keep
working unmodified). The gateway itself draws no distinction between the two at provisioning
time — confirmed against the installed `mlflow` package's gateway handlers
(`mlflow.server.handlers._create_gateway_endpoint` takes no fixed "type"; the unified
`/invocations` handler in `mlflow.server.gateway_api` autodetects chat vs. embeddings per
request from the payload shape, `"messages"` vs `"input"`) — an embeddings route is just this
same three-call flow pointed at a different (embeddings-capable) upstream.

Not idempotent by design — MLflow has no upsert-by-name for these resources. Each route is
provisioned independently: if a route already exists (secret/model-definition/endpoint name
collision), that route is skipped with a warning rather than aborting the whole script — so
re-running after only the chat route exists still provisions a newly-added embeddings route.
Delete a route first via the UI or the /gateway/endpoints/delete API if you need to re-provision
it with different settings.
"""

from __future__ import annotations

import os
from pathlib import Path
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

_session = requests.Session()
if MLFLOW_TRACKING_TOKEN:
    _session.headers["Authorization"] = f"Bearer {MLFLOW_TRACKING_TOKEN}"


_HTTP_BAD_REQUEST = 400


class _RouteAlreadyProvisioned(Exception):
    """Raised when a gateway resource already exists — lets `main()` skip just that route."""


def _post(path: str, body: dict[str, Any], *, route_name: str) -> dict[str, Any]:
    url = f"{MLFLOW_TRACKING_URI}/api/3.0/mlflow/gateway/{path}"
    resp = _session.post(url, json=body, timeout=30)
    if resp.status_code == _HTTP_BAD_REQUEST and "RESOURCE_ALREADY_EXISTS" in resp.text:
        raise _RouteAlreadyProvisioned(route_name)
    if not resp.ok:
        _logger.error("gateway_post_failed", url=url, status_code=resp.status_code, body=resp.text)
        resp.raise_for_status()
    result: dict[str, Any] = resp.json()
    return result


def _provision_route(*, route_name: str, base_url: str, api_key: str, model_name: str) -> None:
    """Create the gateway secret, model definition, and endpoint for one route, in that order.

    Raises `_RouteAlreadyProvisioned` if any of the three resources already exist — the script
    isn't idempotent (MLflow has no upsert-by-name here), so a route that's already provisioned
    is skipped by the caller rather than retried or partially recreated.
    """
    _logger.info("creating_gateway_secret", route=route_name)
    secret = _post(
        "secrets/create",
        {
            "secret_name": f"{route_name}-key",
            # {"api_key": ...} is the shape mlflow's "openai" provider reads for auth —
            # see mlflow.gateway.config._AuthConfigKey.API_KEY.
            "secret_value": {"api_key": api_key},
            "provider": "openai",
            # {"api_base": ...} tells the "openai" provider to call our self-hosted endpoint
            # instead of api.openai.com — see mlflow.server.gateway_api._build_endpoint_config.
            "auth_config": {"api_base": base_url},
        },
        route_name=route_name,
    )
    secret_id = secret["secret"]["secret_id"]
    _logger.info("gateway_secret_created", secret_id=secret_id)

    _logger.info("creating_gateway_model_definition", route=route_name)
    model_definition = _post(
        "model-definitions/create",
        {
            "name": route_name,
            "secret_id": secret_id,
            "provider": "openai",
            "model_name": model_name,
        },
        route_name=route_name,
    )
    model_definition_id = model_definition["model_definition"]["model_definition_id"]
    _logger.info("gateway_model_definition_created", model_definition_id=model_definition_id)

    _logger.info("creating_gateway_endpoint", route=route_name)
    endpoint = _post(
        "endpoints/create",
        {
            "name": route_name,
            "model_configs": [
                {
                    "model_definition_id": model_definition_id,
                    "linkage_type": "PRIMARY",
                    "weight": 1.0,
                }
            ],
        },
        route_name=route_name,
    )
    _logger.info("gateway_endpoint_created", endpoint_id=endpoint["endpoint"]["endpoint_id"])
    _logger.info(
        "done",
        route=route_name,
        # The unified per-route endpoint is mounted directly under /gateway, not under
        # /gateway/mlflow/v1 — see agents_common.models._GatewayEmbeddings's docstring for why
        # that distinction matters (only /gateway/mlflow/v1/chat/completions is a generic route).
        invocations_url=f"{MLFLOW_TRACKING_URI}/gateway/{route_name}/mlflow/invocations",
    )


def _provision_route_or_skip(**kwargs: Any) -> None:
    try:
        _provision_route(**kwargs)
    except _RouteAlreadyProvisioned as exc:
        _logger.warning(
            "gateway_route_already_provisioned",
            route=str(exc),
            hint=(
                "This script isn't idempotent — delete the existing secret/model-definition/"
                "endpoint via the MLflow UI first if you need to re-provision, or leave it "
                "as-is if it's already working."
            ),
        )


def main() -> None:
    """Provision the chat route, plus the embeddings route if its env vars are set.

    Each route is independently skip-if-already-provisioned, so re-running this script after one
    route exists (e.g. only the embeddings route is new) still provisions the others.
    """
    configure_logging()

    chat_model_name = os.environ.get("SELFHOSTED_MODEL_NAME", "gpt-oss-120b")
    _provision_route_or_skip(
        route_name=os.environ.get("GATEWAY_ROUTE_NAME", chat_model_name),
        base_url=os.environ["SELFHOSTED_MODEL_BASE_URL"],
        api_key=os.environ.get("SELFHOSTED_MODEL_API_KEY", "unused"),
        model_name=chat_model_name,
    )

    embedding_base_url = os.environ.get("SELFHOSTED_EMBEDDING_MODEL_BASE_URL", "")
    if not embedding_base_url:
        _logger.warning(
            "embedding_route_skipped",
            reason="SELFHOSTED_EMBEDDING_MODEL_BASE_URL not set",
        )
        return

    embedding_model_name = os.environ.get("SELFHOSTED_EMBEDDING_MODEL_NAME", "")
    _provision_route_or_skip(
        route_name=os.environ.get("EMBEDDING_GATEWAY_ROUTE_NAME", embedding_model_name),
        base_url=embedding_base_url,
        api_key=os.environ.get("SELFHOSTED_EMBEDDING_MODEL_API_KEY", "unused"),
        model_name=embedding_model_name,
    )


if __name__ == "__main__":
    main()
