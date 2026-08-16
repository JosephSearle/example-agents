# supervisor-agent — Tier 2: supervisor pattern (stub)

Not implemented yet — placeholder for a follow-up PR.

**Pattern:** a dedicated supervisor node (LLM + structured output) sits above several
specialist subgraphs and routes to them via hardcoded conditional edges. Matches the
`core-support-agent` architecture described in
`playbook/pages/development/spikes/langgraph-swarm-pattern-itz-15103`.

**Reach for this tier when:** the workflow has a clear top-down chain of command — one
coordinator deciding which specialist handles a request — and you want centralised,
auditable routing logic rather than each agent deciding for itself.

**Planned stack:** `langgraph-supervisor` + raw `StateGraph`, same `agents-common`
checkpointing/observability/config as `react-agent`.
