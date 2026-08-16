# network-mesh-agent — Network / Mesh topology (stub)

Not implemented yet — placeholder for a follow-up PR.

**Pattern:** agents that can all talk to each other, many-to-many, with no fixed hierarchy — any
agent can decide which agent to call next. See
[docs/patterns/agent/network-mesh.md](../../../docs/patterns/agent/network-mesh.md) for the full
writeup.

**Reach for this when:** only once
[Supervisor](../supervisor-agent/README.md)'s hierarchy and
[Swarm / Handoffs](../swarm-agent/README.md)'s peer-handoff both genuinely don't fit. Of the four
multi-agent topologies covered in this doc series, this is the only one without a dedicated,
actively-maintained first-party doc page or standalone library — that's a real signal about the
pattern's actual industry standing, and itself a reason to reach for it last.

**Planned stack:** raw LangGraph `StateGraph` where each agent node's own routing function
decides the next agent, same `agents-common` checkpointing/observability/config as `react-agent`.
