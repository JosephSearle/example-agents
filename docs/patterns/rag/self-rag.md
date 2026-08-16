# Self-RAG

[Corrective RAG](./corrective-rag-crag) checks retrieval quality *before* generating — grading retrieved documents and correcting course if they're weak. Self-RAG goes a step further: it also checks its **own generation** afterward — is this answer actually grounded in what was retrieved, and does it actually address the question — and loops back to fix it if not, rather than trusting the first draft.

![Self-RAG flowchart with retry loop: Question, Retrieve, Grade Documents, Generate, Check Groundedness, Check Usefulness, Answer, with a Rewrite Query loop back to Retrieve on failing grades](../../../public/images/rag/self-rag.svg)

> **Note:**
> This doc assumes a working retriever from [Basic RAG](./basic-rag) / [Collection Creation](../milvus/collection-creation), and reuses the document-relevance grading concept from [Corrective RAG](./corrective-rag-crag) rather than re-explaining it. Like CRAG, this requires LangGraph — but this graph is **cyclic**, not just branching: it can loop back to regeneration or re-retrieval. The official [LangGraph Self-RAG notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb) is now archived upstream ("see the newly consolidated LangChain documentation"), so the walkthrough below reconstructs its well-documented shape from [LangChain's own writeup](https://www.langchain.com/blog/agentic-rag-with-langgraph) and the paper, rather than quoting a live notebook verbatim.

## Core concepts

- **Reflection tokens (paper) → grader chains (practice)** — the [Self-RAG paper](https://arxiv.org/abs/2310.11511) trains a model to emit four kinds of reflection tokens during generation. The practical LangGraph pattern approximates each with a dedicated LLM-as-judge chain instead of training a new model:
  - **Retrieve** (should I fetch passages right now?) ≈ approximated in the Beginner graph by always retrieving once up front; refined in the Advanced section toward a real per-segment decision.
  - **ISREL** (is this passage relevant?) ≈ the same `retrieval_grader` concept already covered in [Corrective RAG](./corrective-rag-crag).
  - **ISSUP** (is this generation supported by the passage?) ≈ a `hallucination_grader`.
  - **ISUSE** (does this generation usefully answer the request?) ≈ an `answer_grader`.
- **Reflection happens after generation, not just before** — this is the key delta from CRAG. CRAG only gates what goes *into* generation; Self-RAG also evaluates what comes *out* of it.
- **The retry loop** — unlike CRAG's one-shot correction, a failed post-generation check can loop back to regenerate (when ungrounded) or to query transformation + re-retrieval (when grounded but unhelpful) — per [selfrag.github.io](https://selfrag.github.io/)'s framing, this is what closes the loop that standard and corrective RAG leave open: neither checks whether the final output actually reflects what was retrieved.

## Implementing the pattern

### Beginner

This follows the practical grader-loop shape documented in LangChain's Self-RAG writeup — the version most devs should start with.

#### 1. Define the graph state
- **Action**: Extend the same `GraphState` shape used in [Corrective RAG](./corrective-rag-crag).
  ```python
  from typing import List
  from typing_extensions import TypedDict

  class GraphState(TypedDict):
      question: str
      generation: str
      documents: List[str]
  ```
- **Result**: The same state shape flows through retrieval, generation, and the new post-generation grading nodes.

#### 2. Retrieve and grade documents
- **Action**: Reuse [Corrective RAG](./corrective-rag-crag)'s `retrieve`, `grade_documents`, and `decide_to_generate` nodes directly — this part of the pipeline doesn't change.
  ```python
  # retrieve, grade_documents, decide_to_generate — same as Corrective RAG
  ```
- **Result**: By the time generation runs, documents have already been filtered for relevance, exactly as in CRAG.

#### 3. Generate
- **Action**: Reuse the same generation step from [Basic RAG](./basic-rag).
  ```python
  def generate(state: GraphState) -> GraphState:
      context = "\n\n".join(doc.page_content for doc in state["documents"])
      generation = llm.invoke(
          f"Answer using only the context below.\n\nContext:\n{context}\n\nQuestion: {state['question']}"
      )
      return {**state, "generation": generation}
  ```
- **Result**: `generation` is a first-draft answer, produced exactly as it would be in Basic RAG or Corrective RAG.

#### 4. Grade the generation
- **Action**: Score the generation two ways — is it grounded in the retrieved documents, and does it actually answer the question?
  ```python
  def grade_generation(state: GraphState) -> dict:
      grounded = hallucination_grader.invoke(
          {"documents": state["documents"], "generation": state["generation"]}
      )
      useful = answer_grader.invoke(
          {"question": state["question"], "generation": state["generation"]}
      )
      return {"grounded": grounded["score"] == "1", "useful": useful["score"] == "1"}
  ```
- **Result**: Two independent boolean judgments — grounding and usefulness — rather than a single pass/fail signal.

#### 5. Route on the result
- **Action**: Define the routing function that decides what happens next based on both judgments.
  ```python
  def grade_generation_v_documents_and_question(state: GraphState) -> str:
      scores = grade_generation(state)
      if not scores["grounded"]:
          return "generate"          # hallucinated — regenerate from the same documents
      if not scores["useful"]:
          return "transform_query"   # grounded, but doesn't answer the question — try again upstream
      return "useful"                # grounded and on-target — done
  ```
- **Result**: The graph branches three ways from a single post-generation check: finish, regenerate, or go back to query transformation and re-retrieval.

#### 6. Assemble the graph
- **Action**: Wire the reused CRAG nodes together with the new generation-grading loop.
  ```python
  from langgraph.graph import StateGraph, END

  workflow = StateGraph(GraphState)
  workflow.add_node("retrieve", retrieve)
  workflow.add_node("grade_documents", grade_documents)
  workflow.add_node("transform_query", transform_query)
  workflow.add_node("generate", generate)

  workflow.set_entry_point("retrieve")
  workflow.add_edge("retrieve", "grade_documents")
  workflow.add_conditional_edges(
      "grade_documents",
      decide_to_generate,
      {"transform_query": "transform_query", "generate": "generate"},
  )
  workflow.add_edge("transform_query", "retrieve")
  workflow.add_conditional_edges(
      "generate",
      grade_generation_v_documents_and_question,
      {"generate": "generate", "transform_query": "transform_query", "useful": END},
  )

  app = workflow.compile()
  ```
- **Result**: `app` is a cyclic graph — a generation that fails grading routes back into the graph instead of returning immediately, up to however many iterations your loop naturally takes.

### Advanced

The Beginner graph always retrieves once, up front, regardless of whether the question actually needs it. The [Self-RAG paper](https://arxiv.org/abs/2310.11511)'s real mechanism is more adaptive: the model decides, per segment, whether retrieval is even necessary before generating that segment. This extension moves the Beginner graph closer to that — treat it as an addition on top of the Beginner section, not something available in the archived reference notebook:

#### 1. Add a retrieval-necessity check
- **Action**: Before retrieving, ask the model whether it can likely answer from its own parametric knowledge, or whether retrieval is actually needed.
  ```python
  def should_retrieve(state: GraphState) -> str:
      decision = retrieve_decision_chain.invoke({"question": state["question"]})
      return "retrieve" if decision["needs_retrieval"] else "generate"
  ```
- **Result**: `should_retrieve` decides, per question, whether to enter the retrieval path at all.

#### 2. Route from the entry point instead of always retrieving
- **Action**: Replace the Beginner graph's fixed entry point with a conditional one.
  ```python
  workflow.set_conditional_entry_point(
      should_retrieve,
      {"retrieve": "retrieve", "generate": "generate"},
  )
  ```
- **Result**: Questions the model judges answerable without retrieval skip straight to `generate`, closer to the paper's per-segment `Retrieve` token than the Beginner graph's always-retrieve-once shortcut.

The tradeoff: this is more faithful to the paper's adaptive retrieval, but adds another LLM judgment call up front and another branch to reason about and debug — most teams should only add this once the Beginner graph's always-retrieve behavior has proven to be a real cost or latency problem.

> **Warning:**
> Common mistakes to check for:
> - **Infinite-loop risk** — a generation that repeatedly fails grading can loop indefinitely. Cap retries with a counter in state and force an exit (e.g. return the best-so-far answer, or an explicit "couldn't produce a grounded answer" response) once the cap is hit.
> - **Extra grading cost** — this adds two more LLM calls per generation attempt (hallucination + answer graders) on top of document grading, and pays that cost again on every loop iteration.
> - **Grader disagreement** — grounded-but-useless and ungrounded-but-on-topic are different failure modes that need different responses (regenerate vs. re-retrieve). Treat `grounded` and `useful` as independent signals, not one combined score.
> - **Web search dependency carries over** — if this is layered on top of [Corrective RAG](./corrective-rag-crag)'s correction branch, the same external-tool-availability caveat from that doc still applies.

## References

- [Self-RAG paper (arXiv:2310.11511)](https://arxiv.org/abs/2310.11511)
- [Self-RAG project page](https://selfrag.github.io/)
- [LangGraph — Self-RAG notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb) (archived — see [LangChain's Self-Reflective RAG writeup](https://www.langchain.com/blog/agentic-rag-with-langgraph) for the current conceptual reference)
- [Basic RAG](./basic-rag)
- [Corrective RAG](./corrective-rag-crag)
- [Milvus — Collection Creation](../milvus/collection-creation)
