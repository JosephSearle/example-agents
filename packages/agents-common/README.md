# agents-common

Shared, non-agent-specific code that every example agent in this repo imports. Exists so that
Postgres checkpointer setup, MLflow tracing setup, and environment-variable config are defined
once — see `docs/decisions/0001-tech-stack.md` for why this is a workspace package instead of
being copy-pasted into each agent.

- `agents_common.config` — `pydantic-settings`-based `Settings`, loaded from the environment
  (see `.env.example` at the repo root).
- `agents_common.checkpointing` — factory functions returning a `PostgresSaver` (short-term,
  per-thread checkpointing) and a `PostgresStore` (long-term, cross-thread memory).
- `agents_common.observability` — one call, `configure_mlflow()`, that points MLflow at the
  tracking server and turns on LangChain/LangGraph autologging.
