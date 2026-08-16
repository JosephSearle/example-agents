# Retrieve & Rerank

Plain vector similarity search is fast but imprecise — a bi-encoder embeds the query and every chunk independently, so the top-k by cosine similarity often includes chunks that are only loosely related. Retrieve & Rerank adds a second, more expensive but far more accurate scoring pass over a small candidate set before anything reaches the LLM: retrieve more candidates than you need, then use a cross-encoder to rerank them and keep only the best.

![Retrieve-then-Rerank pipeline: Question, Vector Search k=20, 20 candidate documents, Cross-Encoder Reranker funnel, 3 surviving documents, Generate, Answer](../../../public/images/rag/retrieve-rerank.svg)

> **Note:**
> This doc assumes you already have a working retriever from [Basic RAG](./basic-rag), pointed at a collection built per [Collection Creation](../milvus/collection-creation). It only covers inserting a rerank stage between retrieval and generation — it doesn't repeat the connect-to-collection or generate steps, and it doesn't cover fine-tuning or training a custom cross-encoder model.

## Core concepts

- **Bi-encoder vs. cross-encoder** — the retrieval step embeds the query and each document independently and compares vectors, which is fast enough to run over an entire collection but only approximates relevance. A cross-encoder instead scores each `(query, document)` pair jointly, which is far more accurate but too slow to run over anything but a small, already-narrowed set of candidates.
- **Over-fetch, then narrow** — retrieve more candidates than you'll actually use (e.g. `k=20`), then rerank down to a smaller `top_n` (e.g. `5`) before generation. The gap between the two numbers is what buys the precision improvement — the cross-encoder gets a chance to correct mistakes the bi-encoder made in its initial ranking.
- **Where it sits in the pipeline** — this slots in directly between the retrieval step and the generation step from [Basic RAG](./basic-rag). Nothing about indexing or generation changes; you're only inserting one stage in the middle.

## Implementing the pattern

The example below uses LangChain's abstractions, starting from a base retriever you already built in [Basic RAG](./basic-rag).

#### 1. Over-fetch a larger candidate set
- **Action**: Build a retriever the same way as in [Basic RAG](./basic-rag), but with a larger `k` than you intend to actually use — you're deliberately over-fetching so the reranker has more to work with.
  ```python
  retriever = vector_store.as_retriever(search_kwargs={"k": 20})
  ```
- **Result**: `retriever` returns 20 candidates per query instead of the final handful you'll actually send to the LLM.

#### 2. Build the cross-encoder reranker
- **Action**: Load a cross-encoder model and wrap it in a reranker that keeps only the top `top_n` results.
  ```python
  from langchain_community.cross_encoders import HuggingFaceCrossEncoder
  from langchain.retrievers.document_compressors import CrossEncoderReranker

  cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
  reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)
  ```
- **Result**: `reranker` can score a list of `(query, document)` pairs and return the 5 most relevant.

#### 3. Wrap the base retriever
- **Action**: Combine the retriever and reranker into a single retriever-shaped object.
  ```python
  from langchain.retrievers import ContextualCompressionRetriever

  compression_retriever = ContextualCompressionRetriever(
      base_compressor=reranker,
      base_retriever=retriever,
  )
  ```
- **Result**: `compression_retriever` behaves like any other retriever, but transparently reranks under the hood.

#### 4. Retrieve + rerank at query time
- **Action**: Call it exactly as you would the base retriever from [Basic RAG](./basic-rag).
  ```python
  reranked_chunks = compression_retriever.invoke(user_question)
  ```
- **Result**: `reranked_chunks` holds the 5 candidates the cross-encoder scored highest out of the original 20 — ready to feed into the same generation step from [Basic RAG](./basic-rag).

> **Warning:**
> Common mistakes to check for:
> - **Over-fetching too aggressively** — the cross-encoder pays a per-document inference cost, so a candidate set of `k=20` is reasonable but `k=1000` will make reranking slow regardless of model size.
> - **`top_n` too small** — cutting too aggressively can discard genuinely relevant chunks that the reranker scored just below the cutoff. Tune `top_n` against your generation step's context budget, not arbitrarily.
> - **Model size vs. latency budget** — small cross-encoders (~22M params) run fine on CPU; larger ones (1.5B+ params) need a GPU to stay fast. Pick a model size that fits your actual latency requirements.
> - **Reranking can't recover a bad first pass** — if the correct chunk wasn't in the initial over-fetched candidate set at all, no amount of reranking will surface it. The first-pass `k` in retrieval still matters.

## References

- [LangChain — Cross-encoder reranker](https://docs.langchain.com/oss/python/integrations/document_transformers/cross_encoder_reranker)
- [Basic RAG](./basic-rag)
- [Milvus — Collection Creation](../milvus/collection-creation)
