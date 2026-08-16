# Basic RAG

Retrieval-augmented generation (RAG) grounds an LLM's answers in your own data — documents, FAQs, code — instead of relying solely on what the model learned during training. Basic RAG is the simplest form of this pattern: on every query, retrieve the most relevant chunks from an already-indexed collection and hand them to the LLM as context before it answers.

![Basic RAG pipeline: Question, Embed Query, Vector Search (top-k), Retrieved Documents, Stuff into Prompt, LLM, Answer](../../../public/images/rag/basic-rag.svg)

> **Note:**
> This doc covers the **single-pass retrieval + generation** pattern only. It does not cover query rewriting, iterative/multi-hop retrieval, re-ranking, or agentic decisions about *whether* or *when* to retrieve — those are separate, more advanced patterns that will get their own docs in this `rag/` section. It also doesn't cover indexing your data — that's a one-time/offline step covered in [Collection Creation](../milvus/collection-creation), and this doc assumes you already have a populated Milvus collection to query.

## Core concepts

- **Retrieval + generation is a request-time operation** — indexing (loading, chunking, embedding, and writing to a collection) is a separate, offline concern handled in [Collection Creation](../milvus/collection-creation). Basic RAG picks up after that: it only runs the live, per-query half of the pipeline.
- **Retrieval `k`** — retrieval returns the top-`k` most similar chunks to the query. `k` trades off completeness against noise: too low and you miss relevant context, too high and you flood the prompt with irrelevant chunks that dilute the model's attention.
- **Grounding** — the retrieved chunks are inserted directly into the LLM's prompt alongside the user's question, so the model answers from what was retrieved rather than from parametric memory. This is what "augmented" means — without it, you'd just be asking the LLM to answer from what it happened to memorize during training.

## Implementing the pattern

The example below uses LangChain's abstractions. It connects straight to an existing Milvus collection — build one first via [Collection Creation](../milvus/collection-creation) (and see [Local Setup](../milvus/setup) if you don't have a Milvus instance to point at yet).

#### 1. Connect to your collection
- **Action**: Point a LangChain vector store at the Milvus collection you already created and populated.
  ```python
  from langchain_milvus import Milvus
  from langchain_openai import OpenAIEmbeddings

  embeddings = OpenAIEmbeddings()
  vector_store = Milvus(
      embedding_function=embeddings,
      collection_name="my_collection",
      connection_args={"uri": "http://localhost:19530"},
  )
  ```
- **Result**: `vector_store` is a LangChain-side handle onto the existing `my_collection` — no data is loaded or written here, it just connects. Use the same embedding model that was used to populate the collection, or similarity search will produce meaningless results.

#### 2. Retrieve at query time
- **Action**: Turn the vector store into a retriever and fetch the top-`k` chunks for an incoming question.
  ```python
  retriever = vector_store.as_retriever(search_kwargs={"k": 4})
  retrieved_chunks = retriever.invoke(user_question)
  ```
- **Result**: `retrieved_chunks` holds the `k` chunks most semantically similar to `user_question`.

#### 3. Generate
- **Action**: Build a prompt combining the retrieved chunks with the user's question, and call the LLM.
  ```python
  from langchain_openai import ChatOpenAI

  context = "\n\n".join(chunk.page_content for chunk in retrieved_chunks)
  prompt = f"""Answer the question using only the context below.

  Context:
  {context}

  Question: {user_question}"""

  llm = ChatOpenAI(model="gpt-4o-mini")
  answer = llm.invoke(prompt)
  ```
- **Result**: `answer` is grounded in the retrieved context rather than the model's unaided parametric knowledge.

> **Warning:**
> Common mistakes to check for:
> - **Embedding mismatch** — the embedding model used at query time must match the one used to populate the collection; otherwise similarity scores are meaningless. See [Collection Creation](../milvus/collection-creation) for where that's decided.
> - **`k` tuned wrong** — too low misses relevant context, too high floods the prompt with noise and drives up token cost/latency without improving answer quality.
> - **Silent failure on empty retrieval** — if no chunk is actually relevant to the question, the model should say so rather than being handed weak context and confidently hallucinating an answer anyway. Handle the empty/low-similarity case explicitly rather than always forwarding whatever `k` chunks came back.

## References

- [LangChain — RAG](https://docs.langchain.com/oss/python/langchain/rag)
- [Milvus — Collection Creation](../milvus/collection-creation)
- [Milvus — Local Setup](../milvus/setup)
