# swarm-agent — Tier 2: swarm pattern (stub)

Not implemented yet — placeholder for a follow-up PR.

**Pattern:** peer specialist agents hand off to each other directly via
`Command(goto=agent_name, update=...)` returned from handover tools (`langgraph-swarm`'s
`create_handoff_tool`) — no central router, no hardcoded inter-agent edges. Matches the
`core-swarm-agent` architecture described in
`playbook/pages/development/spikes/langgraph-swarm-pattern-itz-15103`.

**Reach for this tier when:** the workflow isn't strictly linear or top-down — control needs
to move back and forth between specialists in a way a single supervisor would have to
represent as awkward multi-hop routing.

**Planned stack:** `langgraph-swarm` + raw `StateGraph`, same `agents-common`
checkpointing/observability/config as `react-agent`.
