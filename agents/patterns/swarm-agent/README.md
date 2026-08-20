# swarm-agent — Tier 2: swarm / handoffs pattern

Peer specialist agents hand off to each other directly via `langgraph-swarm`'s
`create_handoff_tool` — no central router, no hardcoded inter-agent edges, and once handed off,
the receiving specialist owns the rest of the conversation entirely. See
[docs/patterns/agent/swarm-handoffs.md](../../../docs/patterns/agent/swarm-handoffs.md) for the
full writeup.

![Swarm / Handoffs topology: three peer agents (Agent A, Agent B, Agent C) arranged in a triangle with direct bidirectional handoff arrows between every pair and no central coordinator; control passes directly from peer to peer, with one agent active at a time](../../../public/images/agent/swarm-handoffs.svg)

**Reach for this when:** the workflow isn't strictly linear or top-down and a specialist should
own the conversation once it takes over — the direct disambiguator from
[Supervisor](../supervisor-agent/README.md): with a handoff, *control moves to the specialist*;
with Supervisor, the main agent stays in charge of the final reply.

## Stack

Unlike [`supervisor-agent`](../supervisor-agent/README.md) — which deliberately avoids
`langgraph-supervisor` per that library's own soft-deprecation notice — this pattern's doc does
**not** steer away from `langgraph-swarm-py`; it's presented as the current LangGraph-ecosystem
equivalent of the handoff pattern. This implementation uses it directly:
`langgraph_swarm.create_handoff_tool` + `create_swarm`. Verified against the current API via the
LangChain reference-docs MCP server (not training-data recollection) — notably, a handoff tool's
only parameters are LangGraph-injected (current state and the tool-call id), so the model never
supplies arguments to trigger a handoff, it just decides *whether* to call it. Same
`agents-common` checkpointing/observability/config wiring as `react-agent`.

## Shape

1. **`triage`** — the default entry point (`DEFAULT_ACTIVE_AGENT`). Its only tool is
   `transfer_to_billing`; it hands off anything billing-related rather than attempting to answer.
2. **`billing`** — the specialist. Tools: `lookup_invoice`, `issue_refund`, plus
   `transfer_to_triage` for anything outside its scope.
3. **`create_swarm([triage_agent, billing_agent], default_active_agent="triage")`** assembles
   both into one graph with persistent "active agent" state: whichever agent last took control is
   remembered across turns in the same thread, so a follow-up message resumes directly with that
   specialist rather than restarting at `triage`. Confirmed empirically (not just per the docs):
   a two-turn run with a canned fake model resumed turn 2 directly with `billing`, with no second
   transfer call needed.
4. Unlike [`supervisor-agent`](../supervisor-agent/README.md) (only the top-level supervisor is
   checkpointed; sub-agents are rebuilt fresh per delegate call), the **entire swarm** — both
   peer agents — is compiled with one checkpointer here, since the active-agent state itself has
   to survive across turns.

## Running it

```bash
make up   # starts Postgres + MLflow + provisions the prompts/dataset/gateway route
uv run swarm-agent "I'd like a refund for invoice INV-1002."
```

Prints whichever agent is currently active's response. Each CLI invocation uses a fresh thread —
to see the handoff persist across turns, invoke `build_swarm` with the same `thread_id` across
multiple calls in a script instead.

## Tests

```bash
make test-unit          # tests/unit — no external services; tools are pure functions
make up
make test-integration   # tests/integration — real Postgres, checkpoint round-tripping
make provision-datasets
make test-eval          # tests/evals — calls a real model via the MLflow AI Gateway
```

- `tests/unit/test_tools.py` — pure-function tests for `lookup_invoice`/`issue_refund`.
- `tests/unit/test_graph.py` — pure logic only (`invoke_config`, `link_prompts_to_trace`'s no-op
  path); building the actual swarm against a stubbed model is exercised at the integration level
  instead, same convention as `react-agent`'s and `supervisor-agent`'s own unit suites.
- `tests/integration/test_checkpointing.py` — a fake `BaseChatModel` plays back a five-call
  sequence across **two turns in the same thread**: triage hands off to billing, billing issues a
  refund; then, without a second handoff call, billing directly answers a follow-up — proving the
  active-agent state actually round-tripped through Postgres.
- `tests/evals/test_quality.py` — MLflow GenAI eval suite scoring whether the expected specialist
  ended up owning the final response (via `create_agent`'s `name=` tag on every message it
  produces), against the seed dataset at `packages/mlflow-server/datasets/swarm-agent.jsonl`.
