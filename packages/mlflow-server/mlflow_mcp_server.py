"""Runs MLflow's built-in MCP server over streamable-http instead of stdio.

`mlflow mcp run` (the CLI both `.mcp.json` and this repo's agent code used before) only speaks
stdio — it's designed to be launched as a subprocess by a single local client (Claude Desktop,
a coding agent), not to be a shared, persistent, network-reachable service. Making `mlflow-mcp`
resolvable via MLflow's MCP Registry (see packages/mlflow-server/scripts/provision_mcp_registry.py)
requires a real HTTP endpoint to register, since `create_mcp_access_endpoint`'s `transport_type`
only accepts `"streamable-http"`/`"sse"`, never `"stdio"` — confirmed against the installed
`mlflow` package, not just its docs.

This is the same MCP server `mlflow mcp run` exposes (`mlflow.mcp.server.create_mcp()`), just
run directly with FastMCP's HTTP transport instead of going through that stdio-only CLI command.
Run as its own docker-compose service (`mlflow-mcp`) alongside the `mlflow` tracking server it
talks to — see docker-compose.yml.
"""

from __future__ import annotations

import os

from mlflow.mcp.server import create_mcp

if __name__ == "__main__":
    # Binding 0.0.0.0 is intentional: this is a containerized service other containers on the
    # compose network need to reach, not a host-facing dev server.
    create_mcp().run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.environ.get("MLFLOW_MCP_PORT", "8001")),
    )
