# Corrective RAG (CRAG)

[Basic RAG](./basic-rag) and [Retrieve & Rerank](./retrieve-rerank) both assume the retrieved context is good enough to generate from. Corrective RAG (CRAG) adds a checkpoint that actually evaluates that assumption — grading retrieved documents for relevance — and corrects course when they fall short, by rewriting the query and falling back to web search rather than generating from context known to be weak.

![Corrective RAG (CRAG) flowchart: Question, Retrieve, Grade Relevance decision, then either Generate to Answer if relevant, or Web Search Fallback merging back into Generate if not relevant](../../../public/images/rag/corrective-rag-crag.svg)

> **Note:**
> This doc assumes a working retriever from [Basic RAG](./basic-rag) / [Collection Creation](../milvus/collection-creation), and can be layered on top of [Retrieve & Rerank](./retrieve-rerank) upstream. Unlike the prior two docs, CRAG is a **stateful graph with conditional branching**, built with LangGraph's `StateGraph` rather than a linear chain — and its correction path depends on a web search tool (the reference implementation uses Tavily) being available.

## Core concepts

- **Retrieval evaluator** — a lightweight grader (an LLM prompted as a relevance classifier) scores each retrieved document against the query, before generation ever happens.
- **Three confidence tiers (paper) vs. binary gate (practice)** — the [CRAG paper](https://arxiv.org/abs/2401.15884) defines three actions based on grading confidence: **Correct** (generate normally), **Incorrect** (discard and fall back to web search), and **Ambiguous** (refine the retrieved context and supplement with web search). The widely-used [LangGraph reference implementation](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag_local.ipynb) simplifies this to a **binary gate**: if any retrieved document is graded irrelevant, correct course; otherwise, generate directly. The binary gate is simpler and covers most real cases — the full 3-tier design gives finer-grained control at the cost of a more complex graph.
- **Query rewriting** — when local retrieval is judged insufficient, the question itself is reformulated before falling back to web search, since the original phrasing may be part of why retrieval failed.
- **Web search as fallback** — when internal retrieval can't be trusted, CRAG supplements or replaces it with live web search results rather than generating from context known to be weak.

## Implementing the pattern

### Beginner

This follows the [LangGraph reference notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag_local.ipynb)'s binary gate — the version most devs should start with.

#### 1. Define the graph state
- **Action**: Declare the state that flows between nodes.
  ```python
  from typing import List
  from typing_extensions import TypedDict

  class GraphState(TypedDict):
      question: str
      generation: str
      documents: List[str]
  ```
- **Result**: `GraphState` is the shared object every node reads from and writes back to.

#### 2. Retrieve
- **Action**: Reuse the retriever from [Basic RAG](./basic-rag) as the first node.
  ```python
  def retrieve(state: GraphState) -> GraphState:
      documents = retriever.invoke(state["question"])
      return {"documents": documents, "question": state["question"]}
  ```
- **Result**: `documents` holds the initial candidate set, exactly as in Basic RAG.

#### 3. Grade documents
- **Action**: Score each retrieved document's relevance with an LLM-as-grader, keeping only those graded relevant.
  ```python
  from langchain_core.output_parsers import JsonOutputParser

  def grade_documents(state: GraphState) -> GraphState:
      relevant_docs = []
      any_irrelevant = False
      for doc in state["documents"]:
          score = grader_chain.invoke({"question": state["question"], "document": doc.page_content})
          if score["score"] == "1":
              relevant_docs.append(doc)
          else:
              any_irrelevant = True
      return {"documents": relevant_docs, "question": state["question"], "web_search_needed": any_irrelevant}
  ```
- **Result**: `documents` is now filtered to only what the grader judged relevant, and `web_search_needed` records whether anything was dropped.

#### 4. Route with a conditional edge
- **Action**: Define `decide_to_generate` — if grading flagged any irrelevant document, route to query transformation; otherwise, go straight to generation.
  ```python
  def decide_to_generate(state: GraphState) -> str:
      return "transform_query" if state["web_search_needed"] else "generate"
  ```
- **Result**: The graph branches based on retrieval quality instead of always generating unconditionally.

#### 5. Transform query and search the web
- **Action**: Rewrite the question, then fall back to web search to supplement the (now filtered) document set.
  ```python
  from langchain_community.tools.tavily_search import TavilySearchResults

  web_search_tool = TavilySearchResults(k=3)

  def transform_query(state: GraphState) -> GraphState:
      better_question = query_rewriter_chain.invoke({"question": state["question"]})
      return {"question": better_question, "documents": state["documents"]}

  def web_search(state: GraphState) -> GraphState:
      results = web_search_tool.invoke({"query": state["question"]})
      state["documents"].extend(results)
      return state
  ```
- **Result**: `documents` is supplemented with fresh web results retrieved using a reformulated, hopefully-better question.

#### 6. Generate
- **Action**: Reuse the same generation step from [Basic RAG](./basic-rag), now fed either the graded-relevant documents or the web-search-augmented set.
  ```python
  def generate(state: GraphState) -> GraphState:
      context = "\n\n".join(doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in state["documents"])
      generation = llm.invoke(f"Answer using only the context below.\n\nContext:\n{context}\n\nQuestion: {state['question']}")
      return {"generation": generation, "documents": state["documents"], "question": state["question"]}
  ```
- **Result**: `generation` is grounded in context that's already been checked for relevance — or corrected when it wasn't.

#### 7. Assemble the graph
- **Action**: Wire the nodes and conditional edge together, then compile.
  ```python
  from langgraph.graph import StateGraph, END

  workflow = StateGraph(GraphState)
  workflow.add_node("retrieve", retrieve)
  workflow.add_node("grade_documents", grade_documents)
  workflow.add_node("transform_query", transform_query)
  workflow.add_node("web_search", web_search)
  workflow.add_node("generate", generate)

  workflow.set_entry_point("retrieve")
  workflow.add_edge("retrieve", "grade_documents")
  workflow.add_conditional_edges(
      "grade_documents",
      decide_to_generate,
      {"transform_query": "transform_query", "generate": "generate"},
  )
  workflow.add_edge("transform_query", "web_search")
  workflow.add_edge("web_search", "generate")
  workflow.add_edge("generate", END)

  app = workflow.compile()
  ```
- **Result**: `app` is a runnable graph — `app.invoke({"question": user_question})` retrieves, grades, corrects if needed, and generates.

### Advanced

The [CRAG paper](https://arxiv.org/abs/2401.15884)'s full design replaces the Beginner section's binary grade with three confidence tiers, adding a refinement path the reference notebook doesn't implement. Treat this as an extension you build on top of the Beginner graph, not something copy-pasteable from the LangGraph reference:

- **Correct** — grading confidence is high; generate directly from the retrieved documents, same as the Beginner section's happy path.
- **Incorrect** — grading confidence is low; discard the retrieved documents entirely and rely on rewritten-query web search alone, rather than mixing in context already judged unreliable.
- **Ambiguous** — grading confidence is mixed; apply **decompose-then-recompose** refinement instead of a blanket keep/discard: split each retrieved document into smaller knowledge strips, grade each strip individually for relevance, discard irrelevant strips, and recompose the relevant ones into a tighter, cleaner context — then supplement that refined context with web search rather than replacing it outright.

Implementing this means replacing `grade_documents`'s binary `{"score": "1"|"0"}` output with a 3-way classification, and adding a new node for strip-level decomposition and filtering that only runs on the Ambiguous branch. The tradeoff versus the Beginner graph: meaningfully finer control over when to trust, refine, or discard retrieved context, at the cost of an extra grading dimension, an extra node, and a harder-to-debug graph.

#### 1. Extend the state
- **Action**: Add a field to track the per-document confidence tier alongside the Beginner section's `GraphState`.
  ```python
  from typing import List, Literal
  from typing_extensions import TypedDict

  class GraphState(TypedDict):
      question: str
      generation: str
      documents: List[str]
      confidence: Literal["correct", "incorrect", "ambiguous"]
  ```
- **Result**: `confidence` carries the tier decision from grading through to routing.

#### 2. Grade with three tiers instead of two
- **Action**: Replace `grade_documents`'s binary score with a 3-way classification per document, then reduce those per-document scores to a single overall tier for the query.
  ```python
  def grade_documents(state: GraphState) -> GraphState:
      tiers = []
      for doc in state["documents"]:
          result = tiered_grader_chain.invoke(
              {"question": state["question"], "document": doc.page_content}
          )
          tiers.append(result["tier"])  # "correct" | "incorrect" | "ambiguous"

      if all(t == "correct" for t in tiers):
          overall = "correct"
      elif all(t == "incorrect" for t in tiers):
          overall = "incorrect"
      else:
          overall = "ambiguous"

      return {**state, "confidence": overall}
  ```
- **Result**: `state["confidence"]` is one of the three paper tiers, replacing the Beginner section's `web_search_needed` boolean.

#### 3. Route on three tiers
- **Action**: Replace `decide_to_generate` with a router that sends each tier down its own path.
  ```python
  def route_by_confidence(state: GraphState) -> str:
      if state["confidence"] == "correct":
          return "generate"
      if state["confidence"] == "incorrect":
          return "web_search_only"
      return "refine_and_supplement"  # ambiguous
  ```
- **Result**: The graph now has three outgoing branches from grading instead of two.

#### 4. Add the decompose-then-recompose refinement node
- **Action**: For the Ambiguous branch, split each document into strips, grade the strips individually, and recompose only the relevant ones.
  ```python
  def refine_and_supplement(state: GraphState) -> GraphState:
      refined_strips = []
      for doc in state["documents"]:
          strips = decompose_into_strips(doc.page_content)  # e.g. split by sentence/paragraph
          for strip in strips:
              score = strip_grader_chain.invoke(
                  {"question": state["question"], "strip": strip}
              )
              if score["score"] == "1":
                  refined_strips.append(strip)

      recomposed_context = " ".join(refined_strips)
      web_results = web_search_tool.invoke({"query": state["question"]})

      return {
          **state,
          "documents": [recomposed_context, *web_results],
      }
  ```
- **Result**: The Ambiguous branch generates from a tighter, strip-filtered version of the original documents plus supplementary web results — rather than either keeping the full (partly irrelevant) documents or discarding them outright.

#### 5. Add an incorrect-only branch
- **Action**: For the Incorrect tier, discard retrieved documents entirely and rely on web search alone.
  ```python
  def web_search_only(state: GraphState) -> GraphState:
      better_question = query_rewriter_chain.invoke({"question": state["question"]})
      web_results = web_search_tool.invoke({"query": better_question})
      return {**state, "question": better_question, "documents": web_results}
  ```
- **Result**: No part of the originally retrieved (untrusted) context reaches generation on this branch.

#### 6. Rewire the graph
- **Action**: Swap the Beginner graph's binary conditional edge for the 3-way router, and wire in the two new nodes.
  ```python
  workflow.add_node("refine_and_supplement", refine_and_supplement)
  workflow.add_node("web_search_only", web_search_only)

  workflow.add_conditional_edges(
      "grade_documents",
      route_by_confidence,
      {
          "generate": "generate",
          "refine_and_supplement": "refine_and_supplement",
          "web_search_only": "web_search_only",
      },
  )
  workflow.add_edge("refine_and_supplement", "generate")
  workflow.add_edge("web_search_only", "generate")
  ```
- **Result**: `app = workflow.compile()` now runs the full 3-tier design — `generate` is reused unchanged from the Beginner section in all three cases.

> **Warning:**
> Common mistakes to check for:
> - **Grading cost** — grading is one extra LLM call per retrieved document, on every query. This adds real latency and cost, and scales with however many documents you retrieve.
> - **Grader miscalibration** — too strict, and you discard good context and over-trigger the (slower, external) web search fallback; too lenient, and irrelevant documents slip through, defeating the point of grading at all.
> - **Web search dependency** — the correction branch depends on an external API/tool being configured and reachable. If that dependency is down or misconfigured, the correction path fails silently unless you handle it explicitly.
> - **Debugging a graph, not a chain** — failures can originate in any node (retrieval, grading, query rewriting, web search, generation). Inspect intermediate state (which documents were graded relevant, what the rewritten query became) rather than only the final answer when something looks wrong.

## References

- [Corrective Retrieval Augmented Generation (CRAG paper)](https://arxiv.org/abs/2401.15884)
- [LangGraph — CRAG reference notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag_local.ipynb)
- [Basic RAG](./basic-rag)
- [Retrieve & Rerank](./retrieve-rerank)
- [Milvus — Collection Creation](../milvus/collection-creation)
