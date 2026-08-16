---
name: new-agent-pattern
description: Scaffold a new agent pattern under agents/patterns/, following this repo's Contributing recipe and the react-agent reference layout. Use when the user wants to add a new pattern (e.g. "add a routing agent", "implement the supervisor pattern", "flesh out the swarm-agent stub").
---

# New agent pattern scaffold

This repo (`example-agents`) ships one fully-implemented reference pattern,
`agents/patterns/react-agent`, and several stubs for the rest. Adding a new pattern (or fleshing
out a stub) follows a fixed recipe from the README's Contributing section and
`docs/decisions/0001-tech-stack.md` (the ADR).

## Steps

1. **Pick and justify a framework tier.** Read the ADR's tiering table
   (`docs/decisions/0001-tech-stack.md`). Ask the user (or infer from the pattern name) which
   tier applies, and be able to state the justification in one sentence:
   - Tier 1 — `langchain.agents.create_agent`: single agent, ReAct loop. Default — use this
     unless there's a concrete reason not to.
   - Tier 2 — raw LangGraph (`StateGraph`, `langgraph-supervisor`, `langgraph-swarm`): multiple
     cooperating agents, or control flow one ReAct loop can't express.
   - Tier 3 — `deepagents`: long-horizon, planning-heavy work with a virtual filesystem. Not for
     anything that fits tiers 1–2.

2. **Scaffold the package.** If `agents/patterns/<name>-agent` already exists as a stub, work
   within it. Otherwise:
   ```bash
   uv init --lib --python 3.12 agents/patterns/<name>-agent
   ```
   Copy structure and conventions from `agents/patterns/react-agent` (the only implemented
   pattern) — `pyproject.toml` shape, `src/` layout, `Dockerfile`, entry point.

3. **Depend on `packages/agents-common`.** Don't reimplement checkpointing, observability, or
   config — pull them from `agents_common`:
   - `agents_common.models.get_chat_model(gateway_route)` for the model (never construct a
     provider client directly).
   - `agents_common.observability` for MLflow tracing setup (`configure_mlflow`).
   - `agents_common.config.Settings` for typed env-based config (Postgres, MLflow, Milvus URIs).
   - Define pattern-local constants the way `react_agent` does: `EXPERIMENT_NAME` and
     `GATEWAY_ROUTE` in the pattern's own package, not shared repo-wide.

4. **Ship tests from day one.** Mirror `agents/patterns/react-agent/tests/`: `tests/unit`
   (no external services), `tests/integration` (needs `make up`, real Postgres), `tests/evals`
   (MLflow eval suite — also add a seed dataset at
   `packages/mlflow-server/datasets/<name>-agent.jsonl` if the pattern needs eval coverage).

5. **Wire it into `docker-compose.yml`.** Add a service block under the `agents` profile,
   modeled on the existing `react-agent` service — same `POSTGRES_*`/`MLFLOW_*` env vars, same
   `depends_on: postgres, mlflow` (both `service_healthy`), same Dockerfile-per-agent build
   context (`agents/patterns/<name>-agent/Dockerfile`).

6. **Update the pattern status table** in `README.md` (the "Patterns and architecture" section) —
   flip the row's status from "Stub" to "Implemented".

## Verification

- `make lint typecheck test-unit` passes for the new package.
- `make test-integration` passes with `make up` running.
- `make demo`-equivalent (run the new agent directly, per its own README) produces a sane
  `AgentResponse`.
