# Query Decomposition

[Basic RAG](./basic-rag) retrieves once per question — fine for simple lookups, but a complex or multi-part question ("how does X compare to Y, and which one should I use for Z?") often can't be answered by any single retrieval pass. Query decomposition breaks the question into smaller sub-questions, retrieves for each separately, and combines the results into a final answer.

![Query Decomposition pipeline: a Complex Question is decomposed into three parallel sub-question branches, each retrieved and answered independently, then converged into a Synthesize node producing the Final Answer](../../../public/images/rag/query-decomposition.svg)

> **Warning:**
> The most commonly cited "official" writeup for this technique, [LangChain's decomposition doc](https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/decomposition/), is pinned to `v0.1` and now redirects to a generic current-docs landing page with no direct replacement. It's cited here for the technique's shape, not as current best-practice documentation. The [`rag-from-scratch`](https://github.com/langchain-ai/rag-from-scratch) notebooks (lessons 5–9, "Query Transformations") are the canonical worked implementation this is grounded in — the notebook itself is large and image-heavy, so treat the code below as a faithful reconstruction of its well-established, widely-cited shape rather than a verbatim quote.

> **Note:**
> This doc assumes an existing retriever from [Basic RAG](./basic-rag) / [Collection Creation](../milvus/collection-creation). It only changes what gets retrieved for, and how many times — nothing about indexing changes.

## Core concepts

- **When decomposition helps (and when it doesn't)** — per the [2025 decomposition paper](https://arxiv.org/pdf/2507.00355), this pattern helps most for genuinely multi-hop or multi-part questions; a simple single-fact question gains nothing from it and just pays extra LLM/retrieval cost for no benefit. Decompose deliberately, not as a default applied to every query.
- **Sub-question generation** — one LLM call turns a complex question into a small set of simpler, independently-retrievable sub-questions.
- **Sequential vs. parallel composition** — two strategies for turning sub-questions into a final answer: **sequential**, where later sub-questions build on earlier answers (good when they genuinely depend on each other), and **parallel**, where every sub-question is answered independently and then combined (simpler, cheaper to parallelize, but can't handle genuine dependencies between sub-questions).
- **Error propagation** — per the paper, a badly-decomposed question or a failed sub-question retrieval doesn't just fail to help — it can actively degrade the final answer compared to not decomposing at all.

## Implementing the pattern

### Sequential

Use this when sub-questions genuinely build on each other — later questions need earlier answers to make sense.

#### 1. Generate sub-questions
- **Action**: Ask an LLM to break the question into an ordered list of simpler sub-questions.
  ```python
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(model="gpt-4o-mini")

  decompose_prompt = f"""Break the following question into a numbered list of
  simpler sub-questions that, answered in order, would help answer the original.

  Question: {user_question}"""

  sub_questions = llm.invoke(decompose_prompt).content.split("\n")
  ```
- **Result**: `sub_questions` is an ordered list, e.g. `["What is X?", "What is Y?", "Given X and Y, which fits Z?"]`.

#### 2. Answer the first sub-question
- **Action**: Retrieve and generate exactly as in [Basic RAG](./basic-rag).
  ```python
  qa_pairs = []

  first_docs = retriever.invoke(sub_questions[0])
  first_context = "\n\n".join(doc.page_content for doc in first_docs)
  first_answer = llm.invoke(
      f"Context:\n{first_context}\n\nQuestion: {sub_questions[0]}"
  ).content

  qa_pairs.append((sub_questions[0], first_answer))
  ```
- **Result**: `qa_pairs` holds the first sub-question/answer pair, grounded in retrieved context.

#### 3. Answer each subsequent sub-question with accumulated context
- **Action**: For every remaining sub-question, retrieve normally, but also feed in the prior Q&A pairs so later answers can build on earlier ones.
  ```python
  for sub_question in sub_questions[1:]:
      docs = retriever.invoke(sub_question)
      context = "\n\n".join(doc.page_content for doc in docs)
      history = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs)

      answer = llm.invoke(
          f"""Prior questions and answers:
      {history}

      New context:
      {context}

      Question: {sub_question}"""
      ).content

      qa_pairs.append((sub_question, answer))
  ```
- **Result**: Each sub-answer can draw on both freshly retrieved context and everything answered so far.

#### 4. Generate the final answer
- **Action**: Synthesize the original question using the full accumulated Q&A chain.
  ```python
  full_history = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs)
  final_answer = llm.invoke(
      f"""Using the sub-questions and answers below, answer the original question.

  {full_history}

  Original question: {user_question}"""
  ).content
  ```
- **Result**: `final_answer` is grounded in a chain of sub-answers that were allowed to build on each other.

### Parallel

Use this when sub-questions are genuinely independent — each can be answered without knowing the others.

#### 1. Generate sub-questions
- **Action**: Same generation step as Sequential — reuse it as-is.
  ```python
  sub_questions = llm.invoke(decompose_prompt).content.split("\n")
  ```
- **Result**: The same ordered list of sub-questions, but this time order doesn't matter for how they're answered.

#### 2. Retrieve and answer each sub-question independently
- **Action**: Answer every sub-question with no shared context between them — each can run concurrently.
  ```python
  def answer_sub_question(sub_question: str) -> tuple[str, str]:
      docs = retriever.invoke(sub_question)
      context = "\n\n".join(doc.page_content for doc in docs)
      answer = llm.invoke(f"Context:\n{context}\n\nQuestion: {sub_question}").content
      return sub_question, answer

  qa_pairs = [answer_sub_question(q) for q in sub_questions]
  ```
- **Result**: `qa_pairs` holds every sub-question's answer, each arrived at independently.

#### 3. Combine all sub-answers
- **Action**: Bring every independent sub-answer together into one block of context.
  ```python
  combined = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs)
  ```
- **Result**: `combined` is a flat summary of everything learned, with no ordering dependency between the pieces.

#### 4. Generate the final answer
- **Action**: Synthesize the original question using all independently-gathered sub-answers.
  ```python
  final_answer = llm.invoke(
      f"""Using the question-answer pairs below, answer the original question.

  {combined}

  Original question: {user_question}"""
  ).content
  ```
- **Result**: `final_answer` is grounded in the union of independently retrieved sub-answers.

> **Warning:**
> Common mistakes to check for:
> - **Decomposing questions that don't need it** — a simple, single-fact question gains nothing from this pattern and just adds latency and cost.
> - **Over-decomposing** — too many trivial sub-questions dilutes focus and multiplies LLM/retrieval calls without improving the answer.
> - **Sequential error propagation** — an early wrong sub-answer poisons every subsequent sub-question that depends on it. Parallel doesn't have this specific failure mode, but can't handle genuinely dependent sub-questions either.
> - **Choosing the wrong strategy** — using parallel for genuinely dependent sub-questions, or sequential for genuinely independent ones, both waste cost and latency without buying anything.

## References

- [Question Decomposition for Retrieval-Augmented Generation (arXiv:2507.00355)](https://arxiv.org/pdf/2507.00355)
- [LangChain — Query decomposition (v0.1, versioned/archived)](https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/decomposition/)
- [`rag-from-scratch` — Query Transformations](https://github.com/langchain-ai/rag-from-scratch)
- [Basic RAG](./basic-rag)
- [Milvus — Collection Creation](../milvus/collection-creation)
