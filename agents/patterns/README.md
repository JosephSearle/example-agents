# agents/patterns/

Reusable reference implementations of agentic patterns, tiered by the framework each one needs —
see [docs/decisions/0001-tech-stack.md](../../docs/decisions/0001-tech-stack.md) for the full
reasoning behind the tiering.

| Pattern | Tier | Status | Path | Doc |
|---|---|---|---|---|
| Single ReAct agent | 1 — `create_agent` | **Implemented** | [`react-agent`](react-agent/README.md) | [react-agent.md](../../docs/patterns/agent/react-agent.md) |
| Prompt Chaining | workflow — raw `StateGraph` | Stub | [`prompt-chaining-agent`](prompt-chaining-agent/README.md) | [prompt-chaining.md](../../docs/patterns/agent/prompt-chaining.md) |
| Routing | workflow — raw `StateGraph` | Stub | [`routing-agent`](routing-agent/README.md) | [routing.md](../../docs/patterns/agent/routing.md) |
| Parallelization | workflow — raw `StateGraph` | Stub | [`parallelization-agent`](parallelization-agent/README.md) | [parallelization.md](../../docs/patterns/agent/parallelization.md) |
| Orchestrator-Workers | workflow — raw `StateGraph` | Stub | [`orchestrator-workers-agent`](orchestrator-workers-agent/README.md) | [orchestrator-workers.md](../../docs/patterns/agent/orchestrator-workers.md) |
| Evaluator-Optimizer | workflow — raw `StateGraph` | Stub | [`evaluator-optimizer-agent`](evaluator-optimizer-agent/README.md) | [evaluator-optimizer.md](../../docs/patterns/agent/evaluator-optimizer.md) |
| Map-Reduce | workflow — raw `StateGraph` | Stub | [`map-reduce-agent`](map-reduce-agent/README.md) | [map-reduce.md](../../docs/patterns/agent/map-reduce.md) |
| Supervisor multi-agent | 2 — LangGraph + `langgraph-supervisor` | Stub | [`supervisor-agent`](supervisor-agent/README.md) | [supervisor.md](../../docs/patterns/agent/supervisor.md) |
| Swarm multi-agent | 2 — LangGraph + `langgraph-swarm` | Stub | [`swarm-agent`](swarm-agent/README.md) | [swarm-handoffs.md](../../docs/patterns/agent/swarm-handoffs.md) |
| Network / Mesh | workflow — raw `StateGraph` | Stub | [`network-mesh-agent`](network-mesh-agent/README.md) | [network-mesh.md](../../docs/patterns/agent/network-mesh.md) |

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
