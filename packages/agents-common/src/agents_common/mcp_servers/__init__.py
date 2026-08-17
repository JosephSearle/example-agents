"""MCP server connections this repo's agent code uses at runtime.

Every server declared in `.mcp.json` (repo root) is resolved dynamically via MLflow's MCP
Registry (https://mlflow.org/docs/latest/genai/mcp-registry/) rather than a hardcoded launch
command — see `_registry_connection` below and
`packages/mlflow-server/scripts/provision_mcp_registry.py`, which registers all four
(`mlflow-mcp`, `milvus-mcp`, `docs-langchain`, `reference-langchain`).

That script only works because `mlflow-mcp`/`milvus-mcp` run as persistent streamable-http
services (`packages/mlflow-server/mlflow_mcp_server.py`, `packages/milvus/milvus_mcp_server.py`, their own
docker-compose services) rather than their stdio-only or localhost-bound CLIs: confirmed against
the installed `mlflow` package that `create_mcp_access_endpoint`'s `transport_type` only accepts
`"streamable-http"`/`"sse"`, never `"stdio"` — a stdio-only (or unreachable) server has nothing
connectable to register. `docs-langchain`/`reference-langchain` needed no such conversion —
they're already remote, third-party-hosted HTTP servers. See docs/decisions/0001-tech-stack.md
for the full investigation.

`milvus_mcp_connection` isn't consumed by any agent yet — see the planned RAG-pattern agent in
README's Roadmap — registered ahead of time so that agent resolves a connection instead of
reintroducing a hardcoded launch command on day one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow
from mlflow.genai.mcp_servers import search_mcp_access_endpoints

if TYPE_CHECKING:
    from langchain_mcp_adapters.sessions import StreamableHttpConnection

    from agents_common.config import Settings

__all__ = [
    "milvus_mcp_connection",
    "mlflow_mcp_connection",
]

# Must match packages/mlflow-server/scripts/provision_mcp_registry.py's `_SERVERS` names —
# MLflow requires a "<reverse-dns namespace>/<slug>" name, a bare slug like "mlflow-mcp" is
# rejected.
_MLFLOW_MCP_SERVER_NAME = "dev.example-agents/mlflow-mcp"
_MILVUS_MCP_SERVER_NAME = "dev.example-agents/milvus-mcp"
_PRODUCTION_ALIAS = "production"

# MLflow's registry uses "streamable-http" (hyphen, matching the MCP spec's transport names);
# langchain-mcp-adapters' StreamableHttpConnection expects "streamable_http" (underscore) — not
# a passthrough, so this mapping is required, not cosmetic.
_TRANSPORT_TYPE_TO_LANGCHAIN = {
    "streamable-http": "streamable_http",
    "sse": "sse",
}


def _registry_connection(server_name: str, *, settings: Settings) -> StreamableHttpConnection:
    """Resolve a server's connection from MLflow's MCP Registry (the `production` alias).

    Raises:
        RuntimeError: No access endpoint is registered — run
            `uv run python packages/mlflow-server/scripts/provision_mcp_registry.py`
            (or `make provision-mcp-registry` / `make up`) first.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    endpoints = search_mcp_access_endpoints(server_name=server_name, server_alias=_PRODUCTION_ALIAS)
    if not endpoints:
        msg = (
            f"'{server_name}' has no registered access endpoint — run "
            "`make provision-mcp-registry` (or `make up`) first."
        )
        raise RuntimeError(msg)

    endpoint = endpoints[0]
    transport = _TRANSPORT_TYPE_TO_LANGCHAIN[endpoint.transport_type.value]
    return {"transport": transport, "url": endpoint.url}  # type: ignore[typeddict-item]


def mlflow_mcp_connection(settings: Settings) -> StreamableHttpConnection:
    """Resolve `mlflow-mcp`'s connection from MLflow's MCP Registry."""
    return _registry_connection(_MLFLOW_MCP_SERVER_NAME, settings=settings)


def milvus_mcp_connection(settings: Settings) -> StreamableHttpConnection:
    """Resolve `milvus-mcp`'s connection from MLflow's MCP Registry."""
    return _registry_connection(_MILVUS_MCP_SERVER_NAME, settings=settings)
