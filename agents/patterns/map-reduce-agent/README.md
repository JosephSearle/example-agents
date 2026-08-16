# map-reduce-agent — Map-Reduce workflow (stub)

Not implemented yet — placeholder for a follow-up PR.

**Pattern:** split work into independent units, process each in parallel (map), then combine the
outputs (reduce) — the same shape as Dean & Ghemawat's original MapReduce, applied to LLM calls
instead of data records. See
[docs/patterns/agent/map-reduce.md](../../../docs/patterns/agent/map-reduce.md) for the full
writeup.

**Reach for this when:** you need [Parallelization](../parallelization-agent/README.md)'s
Sectioning variant generalized — instead of a fixed set of subtasks written by the developer
ahead of time, the number of parallel branches is determined **at runtime** by graph state.

**Planned stack:** raw LangGraph `StateGraph` using the `Send` API for runtime-determined
fan-out, same `agents-common` checkpointing/observability/config as `react-agent`.
