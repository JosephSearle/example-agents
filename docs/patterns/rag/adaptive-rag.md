# Adaptive RAG

Every pattern covered so far in this series commits to one fixed strategy for every query: [Basic RAG](./basic-rag) always retrieves once, [Corrective RAG](./corrective-rag-crag) always grades and potentially corrects, [Query Decomposition](./query-decomposition) always breaks the question apart. Adaptive RAG instead classifies each query first and routes it to whichever strategy actually fits — skipping retrieval entirely for queries an LLM can already answer, using simple retrieval for straightforward ones, and reaching for heavier machinery only when the query genuinely needs it.

![Adaptive RAG: a Question is classified, then routed to one of three difficulty-based branches — no_retrieval answers directly, single_hop retrieves k=3 then generates, multi_hop retrieves k=8 and reasons iteratively before answering](../../../public/images/rag/adaptive-rag.svg)

> **Note:**
> This doc assumes [Basic RAG](./basic-rag)/[Collection Creation](../milvus/collection-creation), and reuses [Corrective RAG](./corrective-rag-crag)'s graph directly as its "retrieval" branch rather than re-explaining it. Like CRAG and Self-RAG, the official [LangGraph Adaptive RAG notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag.ipynb) is archived upstream ("retained purely for archival purposes... see the newly consolidated LangChain documentation") — the walkthrough below reconstructs its well-documented shape rather than quoting a live notebook verbatim.

## Core concepts

- **Route before you retrieve** — the new idea versus every earlier doc in this series: a classification step runs *before* any retrieval decision, deciding whether retrieval should happen at all, and if so, how much of it.
- **Trained classifier (paper) vs. prompted router (practice)** — the [Adaptive-RAG paper](https://arxiv.org/abs/2403.14403) trains a dedicated small model (T5-large, per the [official repo](https://github.com/starsuzi/Adaptive-RAG)) on **pseudo-labels**: every training query is run through all three strategies, and labeled with whichever one actually got it right. The practical LangGraph reference substitutes a single prompted LLM call with structured output instead of training anything. Tradeoff: the prompted router is far cheaper to stand up; a trained classifier is likely more consistent and cheaper per-query at real scale, at the cost of an actual training pipeline — labeling, running all three strategies across a training set, and training a model, not just writing a prompt.
- **No-retrieval as a real branch** — every prior doc in this series assumed retrieval always happens. Adaptive RAG's cheapest branch is "don't retrieve at all," and that's only safe when the classifier is genuinely good at recognizing queries the LLM can answer unaided — a misfire here is the riskiest failure mode in this pattern.

## Implementing the pattern

### Beginner

This follows the archived LangGraph notebook's practical shape: a binary router in front of a reused [Corrective RAG](./corrective-rag-crag) graph.

#### 1. Define the router
- **Action**: Use an LLM with structured output to classify the incoming query as needing the internal knowledge base or the open web.
  ```python
  from pydantic import BaseModel, Field
  from typing import Literal

  class RouteQuery(BaseModel):
      datasource: Literal["vectorstore", "web_search"] = Field(
          description="Route the question to the vectorstore or web search."
      )

  router_llm = llm.with_structured_output(RouteQuery)

  def route_question(state: GraphState) -> str:
      result = router_llm.invoke(state["question"])
      return result.datasource
  ```
- **Result**: `route_question` returns `"vectorstore"` for questions your collection likely covers, or `"web_search"` for anything else.

#### 2. Route from the entry point
- **Action**: Use a conditional entry point instead of a fixed `set_entry_point`, so routing happens before any retrieval.
  ```python
  workflow.set_conditional_entry_point(
      route_question,
      {"vectorstore": "retrieve", "web_search": "web_search"},
  )
  ```
- **Result**: `"vectorstore"` queries enter [Corrective RAG](./corrective-rag-crag)'s graph at `retrieve`; `"web_search"` queries skip straight to `web_search`, bypassing the internal collection entirely.

#### 3. Reuse the Corrective RAG graph downstream
- **Action**: Wire in the same nodes from [Corrective RAG](./corrective-rag-crag) unchanged — `retrieve`, `grade_documents`, `decide_to_generate`, `transform_query`, `web_search`, `generate`.
  ```python
  # retrieve, grade_documents, decide_to_generate,
  # transform_query, web_search, generate — same as Corrective RAG
  ```
- **Result**: Everything after the initial routing decision is exactly the correction-capable graph already documented in [Corrective RAG](./corrective-rag-crag) — nothing new to re-explain here.

#### 4. Assemble the graph
- **Action**: Combine the conditional entry point with the reused CRAG nodes and edges.
  ```python
  from langgraph.graph import StateGraph, END

  workflow = StateGraph(GraphState)
  workflow.add_node("retrieve", retrieve)
  workflow.add_node("grade_documents", grade_documents)
  workflow.add_node("transform_query", transform_query)
  workflow.add_node("web_search", web_search)
  workflow.add_node("generate", generate)

  workflow.set_conditional_entry_point(
      route_question,
      {"vectorstore": "retrieve", "web_search": "web_search"},
  )
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
- **Result**: `app` now decides, per query, whether to even touch your collection before running the rest of the Corrective RAG pipeline.

### Advanced

The Beginner graph only makes a binary source decision (internal vs. web) — it still always retrieves from *somewhere*. The [Adaptive-RAG paper](https://arxiv.org/abs/2403.14403)'s actual design is a 3-way complexity classification that includes skipping retrieval entirely. Treat this as an extension on top of the Beginner graph, not something in the archived notebook:

- **`no_retrieval`** — the query is simple enough for the LLM to answer directly from its own knowledge; skip retrieval entirely.
- **`single_step`** — the existing reused [Corrective RAG](./corrective-rag-crag)/[Basic RAG](./basic-rag) retrieval path is sufficient.
- **`multi_step`** — the query is complex enough to route into a heavier pattern already covered in this series, such as [Query Decomposition](./query-decomposition) or [Self-RAG](./self-rag)'s retry loop.

#### 1. Replace the binary router with a 3-way classifier
- **Action**: Extend `RouteQuery` to a complexity classification instead of a source classification.
  ```python
  class ComplexityRoute(BaseModel):
      complexity: Literal["no_retrieval", "single_step", "multi_step"] = Field(
          description="Classify how much retrieval, if any, this question needs."
      )

  complexity_router = llm.with_structured_output(ComplexityRoute)

  def route_by_complexity(state: GraphState) -> str:
      result = complexity_router.invoke(state["question"])
      return result.complexity
  ```
- **Result**: Queries are now sorted into three tiers instead of two.

#### 2. Add the no-retrieval branch
- **Action**: Answer directly from the LLM's own knowledge, with no retrieval step at all.
  ```python
  def generate_direct(state: GraphState) -> GraphState:
      answer = llm.invoke(state["question"])
      return {**state, "generation": answer}
  ```
- **Result**: The cheapest possible branch — no retrieval, no grading, just a direct answer.

#### 3. Route the three tiers
- **Action**: Wire the 3-way router into the conditional entry point.
  ```python
  workflow.set_conditional_entry_point(
      route_by_complexity,
      {
          "no_retrieval": "generate_direct",
          "single_step": "retrieve",
          "multi_step": "decompose",  # e.g. Query Decomposition's entry node
      },
  )
  ```
- **Result**: Simple queries skip retrieval, moderate queries use the reused single-pass graph, and complex queries are handed off to a heavier pattern already documented elsewhere in this series.

For the most faithful version of this idea, the [official Adaptive-RAG repo](https://github.com/starsuzi/Adaptive-RAG) doesn't prompt an LLM for this classification at all — it trains a dedicated small classifier (T5-large) on pseudo-labels obtained by actually running all three strategies against a training set and recording which one got each query right. That's a real ML training investment (labeled data, a multi-strategy evaluation harness, model training), not a runbook-sized task — worth pursuing only once a prompted router's accuracy has proven to be a real bottleneck.

> **Warning:**
> Common mistakes to check for:
> - **Misrouted no-retrieval** — the single riskiest branch. A query wrongly classified as answerable without retrieval will confidently answer with nothing grounding it, and there's no downstream check to catch that.
> - **Router miscalibration** — too eager to escalate to multi-step wastes cost and latency on queries that didn't need it; too reluctant leaves genuinely complex queries under-served by a single retrieval pass.
> - **The router itself isn't free** — even the "cheap" no-retrieval branch still pays for the classification call that decided to skip retrieval.
> - **A prompted router needs validation, not just trust** — like every LLM-as-judge step in this series, its accuracy is only as good as its prompt (or training data). Validate it against real query examples before relying on the no-retrieval branch in production.

## References

- [Adaptive-RAG paper (arXiv:2403.14403)](https://arxiv.org/abs/2403.14403)
- [Official repo (starsuzi/Adaptive-RAG)](https://github.com/starsuzi/Adaptive-RAG)
- [LangGraph — Adaptive RAG notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag.ipynb) (archived)
- [Basic RAG](./basic-rag)
- [Corrective RAG](./corrective-rag-crag)
- [Self-RAG](./self-rag)
- [Query Decomposition](./query-decomposition)
- [Milvus — Collection Creation](../milvus/collection-creation)
