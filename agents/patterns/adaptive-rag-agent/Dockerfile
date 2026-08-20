# Multi-stage build: install this agent's dependencies with uv, then run it in a slim runtime
# image. Built from the repo root so the workspace lockfile and workspace members are visible:
#   docker build -f agents/patterns/adaptive-rag-agent/Dockerfile -t adaptive-rag-agent .
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Copy the whole workspace: uv needs sibling members (agents-common, basic-rag-agent) to resolve
# the lockfile, even though only adaptive-rag-agent's runtime deps end up installed below.
COPY pyproject.toml uv.lock ./
COPY packages/agents-common ./packages/agents-common
COPY agents/patterns/basic-rag-agent ./agents/patterns/basic-rag-agent
COPY agents/patterns/adaptive-rag-agent ./agents/patterns/adaptive-rag-agent

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package adaptive-rag-agent

FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/agents/patterns/adaptive-rag-agent /app/agents/patterns/adaptive-rag-agent
COPY --from=builder /app/agents/patterns/basic-rag-agent /app/agents/patterns/basic-rag-agent
COPY --from=builder /app/packages/agents-common /app/packages/agents-common
# Only this agent's own judge text files, not the rest of packages/mlflow-server — read at import
# time by adaptive_rag_agent.graph's PRODUCTION_SCORERS via agents_common.judges.load_judge_guidelines.
COPY packages/mlflow-server/judges/adaptive-rag-agent-grounded_in_context.txt ./packages/mlflow-server/judges/adaptive-rag-agent-grounded_in_context.txt
COPY packages/mlflow-server/judges/adaptive-rag-agent-routed_appropriately.txt ./packages/mlflow-server/judges/adaptive-rag-agent-routed_appropriately.txt

ENV PATH="/app/.venv/bin:${PATH}"

RUN useradd --create-home --uid 1000 agent
USER agent

ENTRYPOINT ["adaptive-rag-agent"]
