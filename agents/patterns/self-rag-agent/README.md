# self-rag-agent — Self-RAG workflow

Reuses [corrective-rag-agent](../corrective-rag-agent/README.md)'s document-grading loop, then
also grades its *own generated answer* for groundedness and usefulness — regenerating on
hallucination, re-retrieving with a rewritten question on grounded-but-not-useful. See
[docs/patterns/rag/self-rag.md](../../../docs/patterns/rag/self-rag.md) for the full writeup.

![Self-RAG flowchart with retry loop: Question, Retrieve, Grade Documents, Generate, Check Groundedness, Check Usefulness, Answer, with a Rewrite Query loop back to Retrieve on failing grades](../../../public/images/rag/self-rag.svg)

**Reach for this when:** retrieval is fine but *generation* is still unreliable — hallucinating
despite good context, or technically-grounded-but-not-actually-answering-the-question. Per
[docs/patterns/rag/adoption-strategy.md](../../../docs/patterns/rag/adoption-strategy.md), reach
for this after diagnosing that retrieval quality (corrective-rag-agent's problem) isn't the issue.

Implements the doc's **core** design (document grading + post-generation grading). The **Advanced**
`should_retrieve` conditional entry point (skip retrieval entirely for some questions) is an
intentional non-goal — that's what
[adaptive-rag-agent](../adaptive-rag-agent/README.md)'s `no_retrieval` branch is for.

## Stack

Raw LangGraph `StateGraph`, genuinely **cyclic** (not just conditionally branching, contrast
corrective-rag-agent) — this pattern's own doc calls out "infinite-loop risk" as its headline
caveat, so two **independent** retry caps are load-bearing:

- `MAX_RETRIES` (retrieval-side, shared naming with corrective-rag-agent's own cap) — re-retrieve
  with a rewritten question when the answer is grounded but not useful.
- `MAX_REGENERATE` (generation-side) — regenerate from the same documents when the answer is
  ungrounded (hallucinating).

These are independent because grounded-but-useless and ungrounded are different failure modes
needing different corrective actions — a shared budget would mask which one is actually recurring.
On either cap, the graph terminates with the **best-so-far answer** (logged via a structured
warning), not a fabricated special string — contrast basic-rag-agent's `NO_CONTEXT_ANSWER`, which
fires on a genuinely different condition (zero retrieved documents, not poor-grading-after-N).

**No web search** — inherited from corrective-rag-agent's own deviation (see its README): the
retrieval-side retry rewrites the question and retries against the same Milvus collection, keeping
this repo's one-credential rule intact.

Reuses `basic-rag-agent`'s Milvus collection and `corrective-rag-agent`'s node shapes — duplicated,
not imported, per this repo's "each pattern reads standalone" convention (see
corrective-rag-agent's README for the reasoning, which applies here too).

## Graph shape

1. **`retrieve`** / **`grade_documents`** / **`decide_to_generate`** / **`transform_query`** —
   identical to corrective-rag-agent's retrieval-side loop.
2. **`generate`** — same shape as corrective-rag-agent's.
3. **`grade_generation`** (new) — two structured-output LLM calls: `hallucination_grader`
   (grounded?) and, only if grounded, `answer_grader` (useful?) — skipping the second call when
   already ungrounded saves an LLM call. Increments `regenerate_count` at *detection* time (in
   this node) rather than in a separate corrective node, since there's no distinct node for
   "regenerate" — it's the same `generate` node looped back into.
4. **`grade_generation_v_documents_and_question`** (conditional edge):
   - not grounded, under `MAX_REGENERATE` → `generate` again; cap reached → `END` (best-so-far)
   - grounded but not useful, under `MAX_RETRIES` → `transform_query`; cap reached → `END`
   - grounded and useful → `END`

`build_rag_graph`'s `retriever=`/`prompts=` parameters mirror the other RAG patterns' injection
points for hermetic tests.

## Running it

```bash
make up
uv run self-rag-agent "What framework tier does react-agent use?"
```

## Tests

```bash
make test-unit          # tests/unit — no external services; retriever and every grading/
                         # generation call stubbed
make up
make test-integration   # tests/integration — real Postgres + real Milvus retrieval, every
                         # grader always passes (fake), so the correction loops aren't exercised
make provision-datasets
make test-eval           # tests/evals — real model + real Milvus, grounded_in_context
```

- `tests/unit/test_graph.py` — the four cases this pattern's own "infinite-loop risk" caveat calls
  for: first-pass success, one regenerate loop, one re-retrieve loop, and both caps reached
  (bounded — asserts exact call counts, proving the graph never loops forever).
- `tests/integration/test_checkpointing.py` — asserts state survives a checkpointer rebuild, real
  Milvus retrieval, all grading/generation faked.
- `tests/evals/test_quality.py` — MLflow GenAI eval suite scoring `grounded_in_context` against
  `packages/mlflow-server/datasets/self-rag-agent.jsonl`.
