# agents/patterns/

Reusable reference implementations of agentic patterns, tiered by the framework each one needs —
see [docs/decisions/0001-tech-stack.md](../../docs/decisions/0001-tech-stack.md) for the full
reasoning behind the tiering.

| Pattern | Tier | Status | Path |
|---|---|---|---|
| Single ReAct agent | 1 — `create_agent` | **Implemented** | [`react-agent`](react-agent/README.md) |
| Supervisor multi-agent | 2 — LangGraph + `langgraph-supervisor` | Stub | [`supervisor-agent`](supervisor-agent/README.md) |
| Swarm multi-agent | 2 — LangGraph + `langgraph-swarm` | Stub | [`swarm-agent`](swarm-agent/README.md) |

Each pattern is its own uv workspace member with its own `pyproject.toml`, `src/`, and `tests/`.
Pattern-specific detail (how it works, what it demonstrates, how to run or test it) lives in that
pattern's own README, not here.

## Using a pattern from an example

Don't fork a pattern's code into `agents/examples/*` — depend on the package instead:

```toml
[project]
dependencies = ["react-agent"]

[tool.uv.sources]
react-agent = { workspace = true }
```

See [`agents/examples/README.md`](../examples/README.md) for the full example-package convention.
