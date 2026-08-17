# Exposes mcp-atlassian (Jira/Confluence) over streamable-http. Unlike milvus/mlflow-mcp, this
# needs no wrapper script — mcp-atlassian's own CLI supports `--transport streamable-http`
# with a `--host`/`--port` it actually honors, so the ENTRYPOINT runs it directly.
FROM python:3.12-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir mcp-atlassian

RUN useradd --create-home --uid 1000 atlassian-mcp
USER atlassian-mcp

EXPOSE 8003

ENTRYPOINT ["mcp-atlassian", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8003", "--path", "/mcp"]
