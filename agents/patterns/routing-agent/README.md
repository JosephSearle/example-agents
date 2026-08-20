# routing-agent — Routing workflow

Classifies an input first, then sends it down whichever specialized path actually fits — rather
than handling every input the same way. See
[docs/patterns/agent/routing.md](../../../docs/patterns/agent/routing.md) for the full writeup.

![Routing workflow: an Input Query is classified, then sent down exactly one of three specialist paths (General Questions, Refund Requests, Technical Support), each leading to its own Response; only one path is active per query](../../../public/images/agent/routing.svg)

**Reach for this when:** there are two distinct reasons to reach for it — routing to a
*specialized* prompt/tool for a given category of task, or routing to a *cheaper or stronger
model* purely for cost and performance reasons. It's a **workflow**, not an agent: the set of
possible routes is fixed in code ahead of time — the classifier picks among predefined categories,
it doesn't invent new ones at runtime.

This package demonstrates the **task routing** flavor specifically (as opposed to model-tier
routing): different categories of support ticket genuinely need different handling — a distinct
handler prompt per category — which makes for a more interesting reference implementation than a
one-line "swap the model" branch.

## Stack

Raw LangGraph `StateGraph`, same as [`prompt-chaining-agent`](../prompt-chaining-agent/README.md)
— there's no tool loop or LLM-decided branching here for `langchain.agents.create_agent` to
compile, just a classification step followed by a conditional dispatch. Same `agents-common`
checkpointing/observability/config wiring as `react-agent` and `prompt-chaining-agent`.

## Graph shape

1. **`classify_ticket`** — an LLM call with structured output (`TicketCategory`, a Pydantic model
   with a `Literal["general", "refund", "technical"]` field) picks exactly one category and writes
   it into `RouteState.category`.
2. **A conditional edge** (`add_conditional_edges`) reads `RouteState.category` and dispatches to
   exactly one of three handler nodes — `handle_general`, `handle_refund`, `handle_technical` —
   each running the model against its own specialized prompt, fetched from the MLflow prompt
   registry the same way `prompt-chaining-agent` fetches its per-step prompts (one prompt name per
   category: `routing-agent-general`, `routing-agent-refund`, `routing-agent-technical`).
3. Every handler node edges straight to `END` — only one handler ever runs per invocation.

## Running it

```bash
make up   # starts Postgres + MLflow + provisions prompts/datasets/gateway route
uv run routing-agent "I was charged twice for my subscription this month"
```

Prints the chosen category and the handler's response, e.g. `[refund] ...`.

## Tests

```bash
make test-unit          # tests/unit — no external services; model and prompts are stubbed
make up
make test-integration   # tests/integration — real Postgres, checkpoint round-tripping
make provision-datasets
make test-eval          # tests/evals — calls a real model via the MLflow AI Gateway
```

- `tests/unit/test_graph.py` — pure graph-shape and dispatch logic against a stubbed model
  (monkeypatches `get_chat_model`), mirroring `prompt_chaining_agent`'s unit suite.
- `tests/integration/test_checkpointing.py` — asserts routing state (in particular `category`)
  survives a rebuild of the compiled graph against a real Postgres-backed checkpointer.
- `tests/evals/test_quality.py` — MLflow GenAI eval suite scoring routing accuracy against the
  seed dataset at `packages/mlflow-server/datasets/routing-agent.jsonl`.
