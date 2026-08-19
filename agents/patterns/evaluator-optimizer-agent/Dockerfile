# Multi-stage build: install this agent's dependencies with uv, then run it in a slim runtime
# image. Built from the repo root so the workspace lockfile and workspace members are visible:
#   docker build -f agents/patterns/evaluator-optimizer-agent/Dockerfile -t evaluator-optimizer-agent .
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Copy the whole workspace: uv needs sibling members (agents-common) to resolve the lockfile,
# even though only evaluator-optimizer-agent's runtime deps end up installed below.
COPY pyproject.toml uv.lock ./
COPY packages/agents-common ./packages/agents-common
COPY agents/patterns/evaluator-optimizer-agent ./agents/patterns/evaluator-optimizer-agent

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package evaluator-optimizer-agent

FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/agents/patterns/evaluator-optimizer-agent /app/agents/patterns/evaluator-optimizer-agent
COPY --from=builder /app/packages/agents-common ./packages/agents-common
# Only this agent's own judge text file, not the rest of packages/mlflow-server (datasets,
# prompts, scripts) — read at import time by evaluator_optimizer_agent.graph's
# PRODUCTION_SCORERS via agents_common.judges.load_judge_guidelines. See that module's
# docstring for why this is a plain file copy rather than an MLflow prompt-registry fetch.
COPY packages/mlflow-server/judges/evaluator-optimizer-agent-meets_criteria.txt ./packages/mlflow-server/judges/evaluator-optimizer-agent-meets_criteria.txt

ENV PATH="/app/.venv/bin:${PATH}"

RUN useradd --create-home --uid 1000 agent
USER agent

ENTRYPOINT ["evaluator-optimizer-agent"]
