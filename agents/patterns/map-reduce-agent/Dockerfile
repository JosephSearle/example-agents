# Multi-stage build: install this agent's dependencies with uv, then run it in a slim runtime
# image. Built from the repo root so the workspace lockfile and workspace members are visible:
#   docker build -f agents/patterns/map-reduce-agent/Dockerfile -t map-reduce-agent .
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Copy the whole workspace: uv needs sibling members (agents-common) to resolve the lockfile,
# even though only map-reduce-agent's runtime deps end up installed below.
COPY pyproject.toml uv.lock ./
COPY packages/agents-common ./packages/agents-common
COPY agents/patterns/map-reduce-agent ./agents/patterns/map-reduce-agent

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package map-reduce-agent

FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/agents/patterns/map-reduce-agent /app/agents/patterns/map-reduce-agent
COPY --from=builder /app/packages/agents-common ./packages/agents-common
# Only this agent's own judge text file, not the rest of packages/mlflow-server (datasets,
# prompts, scripts) — read at import time by map_reduce_agent.graph's PRODUCTION_SCORERS via
# agents_common.judges.load_judge_guidelines. See that module's docstring for why this is a
# plain file copy rather than an MLflow prompt-registry fetch.
COPY packages/mlflow-server/judges/map-reduce-agent-relevant_jokes.txt ./packages/mlflow-server/judges/map-reduce-agent-relevant_jokes.txt

ENV PATH="/app/.venv/bin:${PATH}"

RUN useradd --create-home --uid 1000 agent
USER agent

ENTRYPOINT ["map-reduce-agent"]
