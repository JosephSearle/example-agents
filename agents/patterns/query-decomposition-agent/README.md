# query-decomposition-agent — Query Decomposition RAG workflow

Splits a multi-part question into independent sub-questions, answers each one against Milvus, then
synthesizes a final answer from all the sub-answers. See
[docs/patterns/rag/query-decomposition.md](../../../docs/patterns/rag/query-decomposition.md) for
the full writeup.

![Query Decomposition pipeline: a Complex Question is decomposed into three parallel sub-question branches, each retrieved and answered independently, then converged into a Synthesize node producing the Final Answer](../../../public/images/rag/query-decomposition.svg)

**Reach for this when:** a question genuinely has multiple independent parts that a single
retrieve-then-generate pass can't cover well in one shot — e.g. "compare X and Y" questions where
X and Y live in different chunks. Per
[docs/patterns/rag/adoption-strategy.md](../../../docs/patterns/rag/adoption-strategy.md), this is
"Hybrid RAG." Decomposing questions that don't need it wastes cost/latency — this pattern doesn't
help (and can hurt) genuinely simple, single-fact questions.

Implements the doc's **parallel** decomposition strategy only. The **sequential** strategy (each
sub-question sees prior sub-Q&A pairs as context, for genuinely *dependent* multi-hop questions) is
an intentional non-goal — this repo already demonstrates that "carry prior context forward" shape
via [prompt-chaining-agent](../prompt-chaining-agent/README.md) and dependent-work delegation via
[orchestrator-workers-agent](../orchestrator-workers-agent/README.md); a third implementation
wouldn't teach anything new.

## Stack

Raw LangGraph `StateGraph`, same "workflow, not agent" framing as every pattern in this repo, even
though the doc itself has no LangGraph at all (plain sequential/parallel Python control flow).

Sub-questions are answered by **one node looping synchronously** over `sub_questions`, not
LangGraph's `Send` API for dynamic fan-out — [map-reduce-agent](../map-reduce-agent/README.md)
already demonstrates that mechanic in this repo; duplicating it here would blur what this pattern
is meant to teach (decomposition, not fan-out mechanics).

Reuses `basic-rag-agent`'s Milvus collection (`COLLECTION_NAME`) and embeddings route. Per this
repo's convention, node functions are duplicated rather than imported from other RAG patterns —
`adaptive-rag-agent` (built after this pattern) duplicates *this* graph's shape for its multi-step
branch, for the same reason.

## Graph shape

1. **`decompose`** — one structured-output LLM call (`SubQuestions`) splitting the question into
   2-4 sub-questions (`MAX_SUB_QUESTIONS`), or returning it unchanged if already simple.
2. **`answer_sub_questions`** — loops over `sub_questions`, retrieving and generating an answer for
   each independently (empty retrieval per sub-question gets an honest placeholder, not a
   hallucination). `sub_answers` stays index-aligned with `sub_questions`.
3. **`synthesize`** — one final LLM call combining every sub-question/sub-answer pair into a
   coherent answer to the *original* question, not just a concatenation of the parts.

`build_rag_graph`'s `retriever=`/`prompts=` parameters mirror the other RAG patterns' injection
points for hermetic tests.

## Running it

```bash
make up
uv run query-decomposition-agent "What framework tier does react-agent use, and what tier does swarm-agent use?"
```

## Tests

```bash
make test-unit          # tests/unit — no external services; retriever and model (decompose,
                         # per-sub-question generate, synthesize) all stubbed
make up
make test-integration   # tests/integration — real Postgres + real Milvus retrieval per
                         # (fixed) sub-question, decompose/generate/synthesize faked
make provision-datasets
make test-eval           # tests/evals — real model + real Milvus, two guideline judges
```

- `tests/unit/test_graph.py` — covers index-alignment between `sub_questions`/`sub_answers`,
  synthesis actually seeing all pairs, and the empty-retrieval-per-sub-question honest fallback.
- `tests/integration/test_checkpointing.py` — asserts state survives a checkpointer rebuild, real
  retrieval per (fixed) sub-question against the seeded collection, everything else faked.
- `tests/evals/test_quality.py` — scores both `grounded_in_context` and
  `addresses_original_question` against
  `packages/mlflow-server/datasets/query-decomposition-agent.jsonl` — hand-curated multi-part
  questions, not reused from basic-rag-agent's single-fact dataset, since decomposition is the
  whole point here.
