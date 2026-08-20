# example-agents

Reference implementations of the agent patterns this team uses — real, running code, not
diagrams. See [`docs/decisions/0001-tech-stack.md`](docs/decisions/0001-tech-stack.md) (the ADR)
for the full reasoning behind every stack choice below; this file is the quick-reference for
working in the repo day to day.

## Stack

Python 3.12, one **uv workspace** monorepo (`agents/*`, `packages/*`, one lockfile). Local infra
via `make up`: Postgres (checkpointing/memory, `agents` + `mlflow` databases), self-hosted MLflow
at `localhost:5000` (tracing + AI Gateway model access), Milvus + Attu at `localhost:19530` /
`localhost:3000` (vector store — wired up in `agents-common`, not yet used by any agent).

## Framework tiering — pick the tier, don't default to the fanciest one

Start every new pattern as **Tier 1**. Only move up with a concrete reason.

| Tier | Framework | Use for |
|---|---|---|
| 1 | `langchain.agents.create_agent` | A single agent with tools in a ReAct loop. The default — reach for this first, every time. |
| 2 | Raw LangGraph (`StateGraph`, `langgraph-supervisor`, `langgraph-swarm`) | Multiple cooperating agents, or control flow one ReAct loop can't express. |
| 3 | `deepagents` | Long-horizon, planning-heavy work with a virtual filesystem and spawned subagents. Not for anything that fits tiers 1–2. |

`agents/patterns/react-agent` is the one fully-implemented pattern (tier 1) — copy its layout when
scaffolding a new one. Everything else under `agents/patterns/` is a stub.

## Commands

```bash
make sync              # uv sync --all-packages
make lint              # ruff check
make format             # ruff format
make typecheck          # mypy --strict
make test-unit          # -m unit, no external services
make test-integration   # -m integration, needs `make up` (real Postgres)
make test-eval          # -m eval, needs `make provision-datasets`, calls a real model
make test-regression    # -m regression, near-100%-pass code-based-grader suite, required in CI
make test                # unit + integration
make up / make down / make reset   # start / stop / nuke+restart local infra
make demo                # runs react-agent end to end
```

Before opening a PR: `make lint typecheck test-unit` (also gated by CI).

## MCP servers (`.mcp.json`)

- **milvus** — inspect/query the local Milvus vector store (`localhost:19530`) directly instead
  of writing throwaway PyMilvus scripts.
- **mlflow-mcp** — query experiments, runs, and traces on the local MLflow server
  (`localhost:5000`) instead of clicking through the UI.
- **docs-langchain** / **reference-langchain** — current LangChain/LangGraph docs and API
  reference. Prefer these over training data for `create_agent`, `langgraph-supervisor`,
  `langgraph-swarm`, and `deepagents` usage — this repo tracks current APIs, not a fixed version.

## Conventions worth knowing

- Every agent gets its model via `agents_common.models.get_chat_model(gateway_route)` — never
  construct a provider client directly; the MLflow AI Gateway is the only model access path.
- Each agent defines its own `EXPERIMENT_NAME` / `GATEWAY_ROUTE` constants rather than sharing one
  repo-wide value — one experiment per agent in MLflow, not one undifferentiated stream.
- New patterns depend on `packages/agents-common` for checkpointing/observability/config instead
  of reimplementing it, and ship `tests/unit`, `tests/integration`, `tests/evals` from day one.
