# Exposes Milvus operations as MCP tools over streamable-http — see milvus_mcp_server.py for why this
# bypasses mcp-server-milvus's own CLI (its --streamable-http mode hard-codes host="localhost",
# unreachable from other containers).
FROM python:3.12-slim-bookworm

WORKDIR /app

# git is needed for the `pip install git+...` below — mcp-server-milvus isn't published to PyPI.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "git+https://github.com/zilliztech/mcp-server-milvus.git"

COPY milvus_mcp_server.py .

RUN useradd --create-home --uid 1000 milvus-mcp
USER milvus-mcp

EXPOSE 8002

ENTRYPOINT ["python", "milvus_mcp_server.py"]
