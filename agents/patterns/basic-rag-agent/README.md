# basic-rag-agent — Basic RAG workflow

Retrieves the most relevant chunks from a Milvus collection for every incoming question, then
hands them to the LLM as grounding context before it answers. See
[docs/patterns/rag/basic-rag.md](../../../docs/patterns/rag/basic-rag.md) for the full writeup —
the first implemented pattern from this repo's separate `docs/patterns/rag/` series.

![Basic RAG pipeline: Question, Embed Query, Vector Search (top-k), Retrieved Documents, Stuff into Prompt, LLM, Answer](../../../public/images/rag/basic-rag.svg)

**Reach for this when:** you need answers grounded in your own data rather than the model's
parametric memory, and a single retrieve-then-generate pass is enough — no query rewriting,
re-ranking, iterative retrieval, or an LLM decision about *whether* to retrieve at all. Those are
separate, more advanced patterns in the same doc series (`retrieve-rerank`, `corrective-rag-crag`,
`self-rag`, `query-decomposition`, `adaptive-rag`) — none implemented in this repo yet.

## Stack

Raw LangGraph `StateGraph`, same framing as [`routing-agent`](../routing-agent/README.md) and
[`prompt-chaining-agent`](../prompt-chaining-agent/README.md) — retrieval always runs before
generation, on every call, so there's no tool loop or LLM-decided branching for
`langchain.agents.create_agent` to compile. Same `agents-common`
checkpointing/observability/config wiring as every other pattern.

**Two AI Gateway routes, not one**: every other agent in this repo calls a single chat route via
`get_chat_model`. This one also needs embeddings, and there's no embeddings path anywhere else in
the repo — `agents_common.models.get_embeddings` was added alongside `get_chat_model`, and
`packages/mlflow-server/scripts/provision_gateway_route.py` now provisions a *second* gateway
route (`EMBEDDING_GATEWAY_ROUTE` / `SELFHOSTED_EMBEDDING_MODEL_*` in `.env`) for it — same
secret → model-definition → endpoint flow as the chat route, just pointed at a different upstream.
Optional: if you haven't got an embeddings-capable endpoint yet, leave
`SELFHOSTED_EMBEDDING_MODEL_BASE_URL` unset and the provisioning script skips that route (with a
warning) rather than failing.

**basic-rag.md's own referenced indexing docs are broken links** (`../milvus/collection-creation`,
`../milvus/setup` don't exist in this repo) — `packages/milvus/scripts/provision_collections.py`
(below) is this repo's actual, if undocumented-elsewhere, answer to "how do I populate a
collection."

## Graph shape

1. **`retrieve`** — connects a `langchain_milvus.Milvus` vector store to the `basic_rag_agent`
   collection, embeds the question via `EMBEDDING_GATEWAY_ROUTE`, and pulls the top-`k` (default
   4) chunks. Narrows each result to its plain `page_content` string immediately, so nothing
   downstream deals with `Document` objects.
2. **`generate`** — per `basic-rag.md`'s own called-out failure mode ("silent failure on empty
   retrieval"), branches explicitly on an empty retrieval: no chunks means an honest
   `NO_CONTEXT_ANSWER` ("I don't have relevant context to answer that question."), not a
   confidently-hallucinated answer built on an empty context block. Otherwise stuffs the
   retrieved chunks plus the question into the registered generation prompt and calls
   `GATEWAY_ROUTE`.

`build_rag_graph`'s `retriever=` parameter lets a caller (tests) inject any object satisfying the
structural `Retriever` protocol in place of the real Milvus-backed one — the only pattern in this
repo whose unit tests need to fake an external retrieval dependency, not just the chat model.

## The Milvus collection

`packages/milvus/collections/basic-rag-agent.jsonl` is a git-tracked seed corpus: short,
hand-curated passages drawn from this repo's own `docs/patterns/{agent,rag}/*.md` — a genuinely
useful demo ("ask this repo's own RAG agent questions about its own architecture patterns"), not
fabricated trivia. `packages/milvus/scripts/provision_collections.py` seeds every `*.jsonl` under
`packages/milvus/collections/` into a same-named (hyphens → underscores) Milvus collection,
drop-if-exists-then-recreate — simpler than `provision-prompts`'/`provision-datasets`' diff-based
idempotency, appropriate for a demo/test collection rather than versioned production data. Wired
into `make up` as `make provision-milvus-collections`.

## Running it

```bash
make up   # starts Postgres + MLflow + Milvus, provisions prompts/gateway routes/the collection
uv run basic-rag-agent "What's the difference between supervisor and swarm/handoffs in this repo?"
```

Prints the grounded answer.

## Tests

```bash
make test-unit          # tests/unit — no external services; model and retriever are both stubbed
make up
make test-integration   # tests/integration — real Postgres AND real Milvus retrieval against the
                         # seeded collection (the one pattern in this repo needing a second live
                         # service for its integration tests, not just Postgres)
make provision-datasets
make test-eval          # tests/evals — calls a real model + real Milvus via the seeded collection
```

- `tests/unit/test_graph.py` — pure graph-shape and empty-retrieval-branch logic against a
  stubbed chat model *and* a fake retriever (no real Milvus).
- `tests/integration/test_checkpointing.py` — asserts retrieval/answer state survives a rebuild of
  the compiled graph against a real Postgres-backed checkpointer, using a real retriever against
  the seeded `basic_rag_agent` collection (fake chat model only).
- `tests/evals/test_quality.py` — MLflow GenAI eval suite scoring `grounded_in_context` against
  the seed dataset at `packages/mlflow-server/datasets/basic-rag-agent.jsonl` — questions
  answerable from the seeded corpus.
