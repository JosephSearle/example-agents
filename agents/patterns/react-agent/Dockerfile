# Multi-stage build: install this agent's dependencies with uv, then run it in a slim runtime
# image. Built from the repo root so the workspace lockfile and workspace members are visible:
#   docker build -f agents/patterns/react-agent/Dockerfile -t react-agent .
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Copy the whole workspace: uv needs sibling members (agents-common) to resolve the lockfile,
# even though only react-agent's runtime deps end up installed below.
COPY pyproject.toml uv.lock ./
COPY packages/agents-common ./packages/agents-common
COPY agents/patterns/react-agent ./agents/patterns/react-agent

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package react-agent

FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/agents/patterns/react-agent /app/agents/patterns/react-agent
COPY --from=builder /app/packages/agents-common /app/packages/agents-common

ENV PATH="/app/.venv/bin:${PATH}"

RUN useradd --create-home --uid 1000 agent
USER agent

ENTRYPOINT ["react-agent"]
