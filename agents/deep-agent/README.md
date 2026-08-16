# deep-agent — Tier 3: deep agent pattern (stub)

Not implemented yet — placeholder for a follow-up PR.

**Pattern:** a `deepagents`-based harness with built-in planning tool, virtual filesystem, and
subagent spawning, for long-horizon, multi-turn work — the shape of problem investigated in
`playbook/pages/development/spikes/deepagent-itz-19739` (a persistent, webhook-driven PR-review
agent that plans, iterates on generated tests, and follows up proactively over days, not a
single request/response turn).

**Reach for this tier when:** the task is long-horizon and planning-first — not just "has
several steps," which `react-agent` or `supervisor-agent`/`swarm-agent` already cover. Using
`deepagents` for a simple tool-calling agent pays for planning/filesystem/subagent machinery
you don't need — see the tiering rule in `docs/decisions/0001-tech-stack.md`.

**Planned stack:** `deepagents`, same `agents-common` checkpointing/observability/config as
`react-agent`.
