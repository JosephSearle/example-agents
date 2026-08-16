# prompt-chaining-agent — Prompt Chaining workflow (stub)

Not implemented yet — placeholder for a follow-up PR.

**Pattern:** decompose one complex task into a sequence of simpler LLM calls, where each step's
output feeds directly into the next — trading latency (multiple LLM round-trips) for accuracy,
since each individual call is a smaller, more focused task. See
[docs/patterns/agent/prompt-chaining.md](../../../docs/patterns/agent/prompt-chaining.md) for the
full writeup.

**Reach for this when:** a task decomposes cleanly into a fixed sequence of subtasks. It's a
**workflow**, not an agent: the sequence of steps is fixed in code ahead of time — the LLM
doesn't decide what step comes next or whether to loop back. If the task needs dynamic branching
or the LLM deciding its own next step, that's [Routing](../routing-agent/README.md) or an agent,
not this.

**Planned stack:** raw LangGraph `StateGraph` with linear edges and an optional gate check
between steps, same `agents-common` checkpointing/observability/config as `react-agent`.
