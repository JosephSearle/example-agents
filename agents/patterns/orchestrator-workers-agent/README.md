# orchestrator-workers-agent — Orchestrator-Workers workflow (stub)

Not implemented yet — placeholder for a follow-up PR.

**Pattern:** a central LLM (the orchestrator) dynamically breaks down a task into subtasks,
delegates each to a worker LLM, and synthesizes their results into a final answer. See
[docs/patterns/agent/orchestrator-workers.md](../../../docs/patterns/agent/orchestrator-workers.md)
for the full writeup.

**Reach for this when:** you need multiple perspectives on the same task but can't predict in
advance which perspectives would be most valuable. It's a boundary case in Anthropic's own
taxonomy — orchestrate → workers → synthesize is a fixed high-level shape, so it's still framed
as a workflow, but unlike [Parallelization](../parallelization-agent/README.md), the *number and
nature* of subtasks is decided by the orchestrator LLM at runtime, not fixed in code.

**Planned stack:** raw LangGraph `StateGraph` with dynamic subtask fan-out driven by
structured-output orchestrator calls, same `agents-common` checkpointing/observability/config as
`react-agent`.
