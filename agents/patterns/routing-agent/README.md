# routing-agent — Routing workflow (stub)

Not implemented yet — placeholder for a follow-up PR.

**Pattern:** classify an input first, then send it down whichever specialized path actually
fits — rather than handling every input the same way. See
[docs/patterns/agent/routing.md](../../../docs/patterns/agent/routing.md) for the full writeup.

**Reach for this when:** there are two distinct reasons to reach for it — routing to a
*specialized* prompt/tool for a given category of task, or routing to a *cheaper or stronger
model* purely for cost and performance reasons. It's a **workflow**, not an agent: the set of
possible routes is fixed in code ahead of time.

**Planned stack:** raw LangGraph `StateGraph` with a conditional edge dispatching on an LLM
classification, same `agents-common` checkpointing/observability/config as `react-agent`.
