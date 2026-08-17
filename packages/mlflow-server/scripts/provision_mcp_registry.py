"""Register every MCP server this repo declares in `.mcp.json` into MLflow's MCP Registry.

`.mcp.json` (repo root) lists four servers for interactive Claude Code use: `mlflow-mcp`,
`milvus-mcp`, `docs-langchain`, `reference-langchain`. Registering all four here — not just the
ones an agent currently consumes — means the next agent that needs one (e.g. the planned
RAG-pattern agent using Milvus, see README's Roadmap) resolves it via
`mlflow.genai.search_mcp_access_endpoints` instead of hardcoding a launch command, the same way
`experiment_analysis_agent.graph` already does for `mlflow-mcp` via
`agents_common.mcp_servers.mlflow_mcp_connection`.

`mlflow-mcp` and `milvus-mcp` needed converting to persistent streamable-http services first
(see `packages/mlflow-server/mlflow_mcp_server.py` / `packages/milvus/milvus_mcp_server.py`) — MLflow's
`create_mcp_access_endpoint` only accepts `transport_type="streamable-http"`/`"sse"`, never
`"stdio"`, confirmed against the installed `mlflow` package, and both servers' own CLIs are
stdio-only (`mlflow mcp run`) or bind to an unreachable `localhost` in HTTP mode
(`mcp-server-milvus --streamable-http`). `docs-langchain`/`reference-langchain` need no such
conversion — they're already remote, third-party-hosted HTTP servers; registering them here is
just cataloging their existing URLs, not standing up new infrastructure.

For each server, two things get created, both idempotent (safe to re-run):
    1. An MCPServer + MCPServerVersion ("1.0.0"), aliased "production" — the catalog entry.
    2. An MCPAccessEndpoint pinned to that alias (not the version), so a future version bump
       doesn't require updating every consumer.

Run after `make up` (needs `mlflow`, `mlflow-mcp`, and `milvus-mcp` healthy). Tool discovery
(see `_refresh_tools` below) needs `mlflow[mcp]` in the process — not a dependency of any agent
package, since none of them import it, so it's added ephemerally via `--with` rather than
bloating every agent's install:

    uv run --with 'mlflow[mcp]>=3.5.1' python packages/mlflow-server/scripts/provision_mcp_registry.py

(`make provision-mcp-registry` / `make up` already do this.)

Known gap, same shape as this repo's other provisioning scripts against a remote instance (see
provision_gateway_route.py / provision_prompts.py): `mlflow-mcp`/`milvus-mcp`'s registered URLs
are host-reachable (`http://localhost:<port>/mcp`), matching how agent code runs today (`uv run`
on the host, not containerized). If an agent consuming either is ever containerized under the
`agents` compose profile, the registered endpoint would need the compose service name instead
(`http://mlflow-mcp:8001/mcp`, `http://milvus-mcp:8002/mcp`) — not solved here, the same
host/container duality `MLFLOW_TRACKING_URI` already has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agents_common.config import get_settings
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.mcp_servers import (
    create_mcp_access_endpoint,
    get_mcp_server_version_by_alias,
    refresh_mcp_server_version_tools,
    register_mcp_server,
    search_mcp_access_endpoints,
    set_mcp_server_alias,
)

_PRODUCTION_ALIAS = "production"
_SERVER_VERSION = "1.0.0"


@dataclass(frozen=True)
class _ServerSpec:
    # MLflow requires a "<reverse-dns namespace>/<slug>" name — a bare slug like "mlflow-mcp" is
    # rejected.
    name: str
    url: str
    transport_type: Literal["streamable-http", "sse"]
    description: str


_SERVERS = [
    _ServerSpec(
        name="dev.example-agents/mlflow-mcp",
        url="http://localhost:8001/mcp",
        transport_type="streamable-http",
        description=(
            "MLflow's built-in MCP server (traces/experiments/scorers), exposed over "
            "streamable-http via packages/mlflow-server/mlflow_mcp_server.py rather than "
            "`mlflow mcp run`'s stdio-only default."
        ),
    ),
    _ServerSpec(
        name="dev.example-agents/milvus-mcp",
        url="http://localhost:8002/mcp",
        transport_type="streamable-http",
        description=(
            "mcp-server-milvus, exposed over streamable-http via packages/milvus/milvus_mcp_server.py "
            "rather than its own CLI's hard-coded localhost-only bind."
        ),
    ),
    _ServerSpec(
        name="dev.example-agents/docs-langchain",
        url="https://docs.langchain.com/mcp",
        transport_type="streamable-http",
        description="LangChain/LangGraph conceptual docs and how-tos, as MCP tools.",
    ),
    _ServerSpec(
        name="dev.example-agents/reference-langchain",
        url="https://reference.langchain.com/mcp",
        transport_type="streamable-http",
        description="LangChain/LangGraph API reference, as MCP tools.",
    ),
]


def _ensure_server_registered(spec: _ServerSpec) -> str:
    """Register `spec` (version `_SERVER_VERSION`, aliased `production`) if not already.

    Returns the version string either way, so `_refresh_tools` has something to target
    regardless of whether this call registered a new version or found an existing one.
    """
    try:
        version = get_mcp_server_version_by_alias(spec.name, _PRODUCTION_ALIAS)
        print(f"'{spec.name}' already registered (version {version.version}), skipping.")
        return version.version
    except MlflowException:
        pass

    print(f"Registering '{spec.name}' version {_SERVER_VERSION}...")
    version = register_mcp_server(
        server_json={
            "name": spec.name,
            "version": _SERVER_VERSION,
            "description": spec.description,
            "remotes": [{"url": spec.url, "type": spec.transport_type}],
        },
        status="active",
        # Discovered separately via `_refresh_tools`, after the access endpoint exists, rather
        # than relying on register_mcp_server's own best-effort discovery here — keeps the two
        # concerns (creating the catalog entry vs. scraping live tools) independently retriable.
        tools=None,
    )
    set_mcp_server_alias(spec.name, _PRODUCTION_ALIAS, version.version)
    print(f"  registered version {version.version}, aliased '{_PRODUCTION_ALIAS}' -> it")
    return version.version


def _ensure_access_endpoint(spec: _ServerSpec) -> None:
    """Create an alias-pinned access endpoint for `spec` if one doesn't already exist."""
    existing = search_mcp_access_endpoints(server_name=spec.name, server_alias=_PRODUCTION_ALIAS)
    if existing:
        print(f"Access endpoint for '{spec.name}'@{_PRODUCTION_ALIAS} already exists, skipping.")
        return

    print(f"Creating access endpoint for '{spec.name}'@{_PRODUCTION_ALIAS}...")
    endpoint = create_mcp_access_endpoint(
        server_name=spec.name,
        url=spec.url,
        transport_type=spec.transport_type,
        server_alias=_PRODUCTION_ALIAS,
    )
    print(f"  endpoint id={endpoint.id} -> {endpoint.url}")


def _refresh_tools(spec: _ServerSpec, version: str) -> None:
    """Scrape live tools from `spec`'s remote and store the snapshot on `version`.

    Unconditional (not skipped for already-registered servers) so re-running this script also
    refreshes a stale/empty tools snapshot — e.g. from a version registered before this step
    existed, or before its remote's tool list changed. Needs `mlflow[mcp]` in this process (see
    the module docstring) — without it, this raises rather than silently storing `tools=None`
    again, since the whole point of calling it is to get tools populated.
    """
    print(f"Refreshing tools for '{spec.name}' version {version}...")
    updated = refresh_mcp_server_version_tools(name=spec.name, version=version)
    print(f"  discovered {len(updated.tools or [])} tool(s)")


def main() -> None:
    """Register every server in `_SERVERS`, its access endpoint, and its tools snapshot."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    for spec in _SERVERS:
        version = _ensure_server_registered(spec)
        _ensure_access_endpoint(spec)
        _refresh_tools(spec, version)

    print("\nDone.")


if __name__ == "__main__":
    main()
