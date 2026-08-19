# orchestrator-workers-agent — Orchestrator-Workers workflow

A central LLM (the orchestrator) dynamically breaks down a task into subtasks, delegates each to
a worker LLM, and synthesizes their results into a final answer. See
[docs/patterns/agent/orchestrator-workers.md](../../../docs/patterns/agent/orchestrator-workers.md)
for the full writeup.

**Reach for this when:** you need multiple perspectives on the same task but can't predict in
advance which perspectives would be most valuable. It's a boundary case in Anthropic's own
taxonomy — orchestrate → workers → synthesize is a fixed high-level shape, so it's still framed
as a workflow, but unlike [Parallelization](../parallelization-agent/README.md), the *number and
nature* of subtasks is decided by the orchestrator LLM at runtime, not fixed in code.

## Stack

Raw LangGraph `StateGraph`, same as [`map-reduce-agent`](../map-reduce-agent/README.md) — there's
no tool loop here for `langchain.agents.create_agent` to compile. Fan-out to workers uses the same
`Send` API map-reduce uses, but goes one step further: map-reduce's `Send` count is
runtime-determined by input list length, while here *both* the count and the content of each
`Send`'s payload come from an LLM call (the orchestrator's structured-output `TaskBreakdown`), not
from a list the caller passed in. Same `agents-common` checkpointing/observability/config wiring
as `react-agent` and `map-reduce-agent`.

The docs page recommends model-tiering (a stronger model for orchestration/synthesis, a cheaper
model for workers) as a cost lever. This reference implementation reuses one `GATEWAY_ROUTE` for
all three stages, same as every other pattern in this repo — a real deployment following this
pattern would pass a second, cheaper `gateway_route` into the worker call.

## Graph shape

1. **`orchestrate`** — one LLM call with structured output (`TaskBreakdown`: an `analysis` string
   plus a `list[Subtask]`, each just a `description`) decomposes the input `task` into however
   many subtasks actually fit it — not a fixed count.
2. **A conditional edge** (`add_conditional_edges`) reads `OverallState.subtasks` and returns one
   `Send("run_worker", {"subtask": ...})` per subtask — dynamic fan-out, exactly like
   `map-reduce-agent`'s `continue_to_jokes`, except the list being fanned out over was itself just
   decided by an LLM call rather than supplied by the caller.
3. **`run_worker`** — each `Send`-spawned worker only sees its own `subtask` (`WorkerState`, not
   the full `OverallState`), keeping workers independent. Writes one result into
   `OverallState.worker_results`, an `Annotated[list[str], operator.add]` reducer field.
4. **`synthesize`** — unlike `map-reduce-agent`'s code-only `combine_jokes`, this fan-in step is
   itself an LLM call: it combines every worker's result into one coherent answer to the original
   task, only firing once every dynamically-spawned worker has completed.

## Running it

```bash
make up   # starts Postgres + MLflow + provisions the prompts/dataset/gateway route
uv run orchestrator-workers-agent "Add rate limiting to a public API — figure out what needs to change."
```

Prints the synthesized final answer — the number of subtasks it took to get there varies by task,
with no change to the graph itself.

## Tests

```bash
make test-unit          # tests/unit — no external services; model is stubbed
make up
make test-integration   # tests/integration — real Postgres, checkpoint round-tripping
make provision-datasets
make test-eval          # tests/evals — calls a real model via the MLflow AI Gateway
```

- `tests/unit/test_graph.py` — asserts worker count tracks the (stubbed) orchestrator's decided
  subtask count exactly, that each worker only ever sees its own subtask, and that `synthesize`
  runs exactly once, after every worker.
- `tests/integration/test_checkpointing.py` — asserts the accumulated `worker_results` and final
  `synthesis` survive a rebuild of the compiled graph against a real Postgres-backed checkpointer.
- `tests/evals/test_quality.py` — MLflow GenAI eval suite scoring whether the orchestrator meets a
  minimum subtask count and whether the synthesis is coherent, against the seed dataset at
  `packages/mlflow-server/datasets/orchestrator-workers-agent.jsonl`. Unlike routing-agent's
  exact-match `correct_category` or map-reduce-agent's exact-match `correct_joke_count`, subtask
  count here has no single ground truth to match — the eval checks a lower bound instead.
