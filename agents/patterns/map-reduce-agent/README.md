# map-reduce-agent — Map-Reduce workflow

Splits work into independent units, processes each in parallel (map), then combines the outputs
(reduce) — the same shape as Dean & Ghemawat's original MapReduce, applied to LLM calls instead of
data records. See [docs/patterns/agent/map-reduce.md](../../../docs/patterns/agent/map-reduce.md)
for the full writeup.

![Map-Reduce pipeline: a Coordinator runs a Map step that fans out to four identical workers each processing a data chunk in parallel, then all four converge and combine into a single Reduce node producing the Merged Result](../../../public/images/agent/map-reduce.svg)

**Reach for this when:** you need [Parallelization](../parallelization-agent/README.md)'s
Sectioning variant generalized — instead of a fixed set of subtasks written by the developer ahead
of time, the number of parallel branches is determined **at runtime** by graph state.

## Stack

Raw LangGraph `StateGraph`, same as [`parallelization-agent`](../parallelization-agent/README.md)
— there's no tool loop or LLM-decided branching here for `langchain.agents.create_agent` to
compile. Unlike sectioning's fixed `for section in SECTIONS: graph.add_edge(START, section)`
loop, fan-out here goes through LangGraph's `Send` API: a routing function returns one `Send` per
input item, however many that turns out to be at invoke time. Same `agents-common`
checkpointing/observability/config wiring as `react-agent` and `parallelization-agent`.

## Graph shape

The worked example generates one joke per input topic, however many topics are given:

1. **`continue_to_jokes`** (the routing function on a conditional edge out of `START`) reads
   `OverallState.topics` and returns `[Send("generate_joke", {"topic": t}) for t in topics]` — one
   `Send` per topic. The graph's structure never names a topic or a count; `len(topics)` at
   invoke time is what determines how many `generate_joke` workers actually run.
2. **`generate_joke`** — each `Send`-spawned worker only sees its own `topic` (`JokeState`, not
   the full `OverallState`), keeping workers independent. Writes one joke into
   `OverallState.jokes`, an `Annotated[list[str], operator.add]` reducer field — every worker
   writes the same key, so the reducer is what accumulates all of their outputs instead of the
   last writer clobbering the rest.
3. **`combine_jokes`** — a code-only fan-in node (no model call) that only fires once every
   dynamically-spawned worker has completed, formatting the jokes into one numbered summary.

## Running it

```bash
make up   # starts Postgres + MLflow + provisions the prompt/dataset/gateway route
uv run map-reduce-agent "cats" "airports" "Mondays"
```

Prints a numbered summary — the number of jokes always matches the number of topic arguments
given, with no change to the graph itself.

## Tests

```bash
make test-unit          # tests/unit — no external services; model is stubbed
make up
make test-integration   # tests/integration — real Postgres, checkpoint round-tripping
make provision-datasets
make test-eval          # tests/evals — calls a real model via the MLflow AI Gateway
```

- `tests/unit/test_graph.py` — asserts the worker call count (and `jokes` length) tracks
  `len(topics)` exactly, across zero/one/several topics, against a stubbed model (monkeypatches
  `get_chat_model`); also asserts each worker only ever sees its own topic.
- `tests/integration/test_checkpointing.py` — asserts the accumulated `jokes` list and combined
  `summary` survive a rebuild of the compiled graph against a real Postgres-backed checkpointer.
- `tests/evals/test_quality.py` — MLflow GenAI eval suite scoring fan-out correctness (does the
  joke count match the topic count) against the seed dataset at
  `packages/mlflow-server/datasets/map-reduce-agent.jsonl`.
