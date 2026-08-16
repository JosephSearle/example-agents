# evaluator-optimizer-agent — Evaluator-Optimizer workflow (stub)

Not implemented yet — placeholder for a follow-up PR.

**Pattern:** one LLM generates a response while another evaluates it and provides feedback, in a
loop — the same iterative process a human writer goes through drafting, getting feedback, and
revising until the piece is polished. See
[docs/patterns/agent/evaluator-optimizer.md](../../../docs/patterns/agent/evaluator-optimizer.md)
for the full writeup.

**Reach for this when:** there are clear evaluation criteria and iterative refinement provides
measurable value — specifically when LLM responses can be demonstrably improved by feedback, and
the LLM can meaningfully provide that feedback on its own output. If neither holds, this pattern
just adds latency and cost for no measurable gain.

**Planned stack:** raw LangGraph `StateGraph` with a generate → evaluate → (loop back or exit)
cycle, capped at a max iteration count, same `agents-common`
checkpointing/observability/config as `react-agent`.
