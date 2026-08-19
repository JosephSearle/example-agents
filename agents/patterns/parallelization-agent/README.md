# parallelization-agent — Parallelization workflow

Runs multiple LLM calls at the same time instead of one after another, then combines their
outputs programmatically — either **Sectioning** (independent pieces of work run concurrently,
for speed) or **Voting** (multiple attempts at the same work, aggregated into one answer, for
confidence). See
[docs/patterns/agent/parallelization.md](../../../docs/patterns/agent/parallelization.md) for the
full writeup.

**Reach for this when:** you can split a task into independent pieces that don't depend on each
other's output, or want multiple independent attempts at the same task to aggregate for
confidence. It's a **workflow**, not an agent: which calls run and how their outputs are combined
is fixed in code ahead of time.

## Stack

Raw LangGraph `StateGraph`, same as [`routing-agent`](../routing-agent/README.md) — there's no
tool loop or LLM-decided branching here for `langchain.agents.create_agent` to compile. Static
edges (not the `Send` API) express the fan-out: the number and identity of sections/voters is
fixed in code, not runtime-determined — `Send`-based fan-out is reserved for
[map-reduce-agent](../map-reduce-agent/README.md)'s runtime-determined case. Same `agents-common`
checkpointing/observability/config wiring as `react-agent` and `routing-agent`.

## Graph shapes

### Sectioning — incident report triage (`build_sectioning_graph`)

An incident description fans out into three independent nodes, all running off `START`:

1. **`summarize`** — one-paragraph plain-language summary.
2. **`assess_severity`** — structured output (`SeverityAssessment`, a Pydantic model with a
   `Literal["low", "medium", "high", "critical"]` field).
3. **`extract_action_items`** — structured output, a bullet list of concrete follow-up actions.

Each section's instruction prompt is fetched from the MLflow prompt registry the same way
`routing-agent` fetches its per-category handler prompts — one prompt name per section
(`parallelization-agent-summarize`, `parallelization-agent-assess_severity`,
`parallelization-agent-extract_action_items`), registered from
`packages/mlflow-server/prompts/parallelization-agent/*.txt` via `make provision-prompts`.

None of the three depends on another's output. All three edge into **`aggregate_report`**, a
code-only fan-in node (no model call) that formats the three fields into one fixed-template
report. LangGraph runs `aggregate_report` automatically once all three predecessors finish for
that superstep — no `Send`/`Command` needed, since each section writes to its own state key and
there's no concurrent-write conflict to resolve with a reducer.

### Voting (`build_voting_graph`)

`n` (default 3) statically-defined `vote_n` nodes each call the model once against the same
prompt and append their attempt into `VoteState.attempts` — an `Annotated[list[str],
operator.add]` field, since every voter writes to the *same* key and needs the reducer to
accumulate rather than clobber. `aggregate_votes` then picks the majority verdict
(`collections.Counter`-style), deliberately kept judge-free so this graph's only nondeterminism is
in the `n` voter calls themselves.

## Running it

```bash
make up   # starts Postgres + MLflow + provisions datasets/gateway route
uv run parallelization-agent "The primary database ran out of disk space and all writes are failing."
uv run parallelization-agent --voting "Is this code safe to merge?"
```

Prints the aggregated report (sectioning) or the majority verdict (voting).

## Tests

```bash
make test-unit          # tests/unit — no external services; model is stubbed
make up
make test-integration   # tests/integration — real Postgres, checkpoint round-tripping
make provision-datasets
make test-eval          # tests/evals — calls a real model via the MLflow AI Gateway
```

- `tests/unit/test_graph.py` — pure graph-shape and aggregation logic against a stubbed model
  (monkeypatches `get_chat_model`), covering both the sectioning and voting graphs.
- `tests/integration/test_checkpointing.py` — asserts sectioning state (in particular `severity`
  and the aggregated `report`) survives a rebuild of the compiled graph against a real
  Postgres-backed checkpointer.
- `tests/evals/test_quality.py` — MLflow GenAI eval suite scoring severity-assessment accuracy
  against the seed dataset at `packages/mlflow-server/datasets/parallelization-agent.jsonl`.
