"""Runs mcp-server-milvus over streamable-http, bound to 0.0.0.0 for container reachability.

`mcp-server-milvus`'s own `--streamable-http` CLI mode hard-codes `mcp.settings.host =
"localhost"` (see zilliztech/mcp-server-milvus's `server.py`, `main()`) with no CLI flag or env
var to override it — unreachable from any other container, including this repo's own compose
network. This bypasses that CLI entirely: imports the same module-level `FastMCP` instance
directly and runs it with `host="0.0.0.0"` instead, replicating the `mcp.config` assignment
`main()` would otherwise do. Same technique `packages/mlflow-server/mlflow_mcp_server.py` uses to work
around `mlflow mcp run`'s stdio-only CLI.
"""

from __future__ import annotations

import os

# `mcp-server-milvus` is only installed inside this file's own Docker image (see
# packages/milvus/Dockerfile) — a git dependency, not part of the main uv workspace venv mypy
# checks against — so this import is unresolvable to mypy outside that container.
from mcp_server_milvus.server import mcp  # type: ignore[import-not-found]

if __name__ == "__main__":
    mcp.config = {
        "milvus_uri": os.environ.get("MILVUS_URI", "http://localhost:19530"),
        "milvus_token": os.environ.get("MILVUS_TOKEN", ""),
        "db_name": os.environ.get("MILVUS_DB", ""),
    }
    mcp.settings.port = int(os.environ.get("MILVUS_MCP_PORT", "8002"))
    # Binding 0.0.0.0 is intentional: this is a containerized service other containers on the
    # compose network need to reach, not a host-facing dev server.
    mcp.settings.host = "0.0.0.0"
    mcp.run(transport="streamable-http")
