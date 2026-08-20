# supervisor-agent — Tier 2: supervisor pattern

A central supervisor agent delegates to specialized sub-agents, each scoped to its own domain and
tools, while the supervisor itself never touches low-level tools directly — it only calls
sub-agents as tools and coordinates the overall response. See
[docs/patterns/agent/supervisor.md](../../../docs/patterns/agent/supervisor.md) for the full
writeup.

![Supervisor topology: a central Supervisor coordinator connected by bidirectional arrows to three workers (Researcher, Analyst, Writer), with no direct links between the workers; all control flow returns through the Supervisor before reaching Output](../../../public/images/agent/supervisor.svg)

**Reach for this when:** you have multiple distinct domains, each with multiple tools or complex
logic, you want centralized workflow control, and sub-agents don't need to converse with users
directly. For simpler cases with just a few tools, a single agent (`react-agent`) is enough. If
sub-agents need to talk to the user directly, that's Swarm/Handoffs (`swarm-agent`, not yet
implemented in this repo), not this pattern.

## Stack

**Deliberately does not depend on the `langgraph-supervisor` package** the ADR's Tier 2 row names
as an example. That library's own README now says: *"We now recommend using the supervisor
pattern directly via tools rather than this library for most use cases."* This implementation
follows that current guidance instead: two full `langchain.agents.create_agent()` sub-agents
(`math`, `text`), each wrapped as an `@tool`-decorated `delegate_to_*` function, and the
supervisor is itself a third `create_agent()` whose only tools are those two delegate wrappers.
No raw `StateGraph`, no extra multi-agent-framework dependency — just `create_agent` composed
twice over. Still genuinely Tier 2 in spirit (multiple cooperating agents, a shape one ReAct loop
can't express), just implemented with the same building block `react-agent` uses rather than a
dedicated supervisor library. Same `agents-common` checkpointing/observability/config wiring as
`react-agent`.

## Shape

1. **`math` sub-agent** — `create_agent` with one tool, `calculator` (restricted-AST arithmetic,
   same implementation as `react-agent`'s, kept local to this package rather than imported —
   every pattern in this repo depends only on `agents-common`, never on a sibling pattern).
2. **`text` sub-agent** — `create_agent` with two tools, `count_words` and `reverse_text`.
3. **`delegate_to_math` / `delegate_to_text`** — thin `@tool`-decorated wrappers, each invoking
   its sub-agent with a fresh, stateless call and returning the sub-agent's final message
   content. From the supervisor's point of view these look exactly like any other tool — the
   sub-agent's internal ReAct reasoning is hidden.
4. **`supervisor`** — a `create_agent()` whose only tools are `delegate_to_math` and
   `delegate_to_text`. This is the only agent that's checkpointed — sub-agents are rebuilt fresh
   on every delegate call and never see a checkpointer, matching
   docs/patterns/agent/supervisor.md's own worked example.

Only the supervisor ever talks to the caller; a request spanning both domains (e.g. "what's 10 +
5, and how many words are in this sentence?") gets delegated to both sub-agents and synthesized
into one answer, without either sub-agent knowing the other exists.

## Running it

```bash
make up   # starts Postgres + MLflow + provisions the prompts/dataset/gateway route
uv run supervisor-agent "What is 10 plus 5, and how many words are in 'hello there friend'?"
```

Prints the supervisor's final synthesized answer.

## Tests

```bash
make test-unit          # tests/unit — no external services; tools are pure functions
make up
make test-integration   # tests/integration — real Postgres, checkpoint round-tripping
make provision-datasets
make test-eval          # tests/evals — calls a real model via the MLflow AI Gateway
```

- `tests/unit/test_tools.py` — pure-function tests for `calculator`/`count_words`/`reverse_text`,
  same style as `react-agent`'s tool tests.
- `tests/unit/test_graph.py` — pure logic only (`invoke_config`, `link_prompts_to_trace`'s no-op
  path); building the actual supervisor against a stubbed model is exercised at the integration
  level instead, same convention as `react-agent`'s own unit suite.
- `tests/integration/test_checkpointing.py` — a fake `BaseChatModel` plays back a fixed
  four-call sequence (supervisor delegates to math → math calls `calculator` → math answers →
  supervisor gives its final answer) to exercise a full supervisor → sub-agent → tool → supervisor
  round trip against real Postgres, without any network access to a live model.
- `tests/evals/test_quality.py` — MLflow GenAI eval suite scoring whether the supervisor
  delegated to every sub-agent a task genuinely required, against the seed dataset at
  `packages/mlflow-server/datasets/supervisor-agent.jsonl`.
