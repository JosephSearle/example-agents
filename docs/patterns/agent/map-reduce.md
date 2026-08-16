# Map-Reduce

This pattern's name is borrowed wholesale from 2000s-era distributed data processing, not from any LLM or agent research. [Dean & Ghemawat's "MapReduce: Simplified Data Processing on Large Clusters" (OSDI 2004)](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/) describes it plainly: *"Users specify a map function that processes a key/value pair to generate a set of intermediate key/value pairs, and a reduce function that merges all intermediate values associated with the same intermediate key."* Split work into independent units, process each in parallel (map), then combine the outputs (reduce) — the exact same shape, just applied to LLM calls instead of data records across a cluster.

![Map-Reduce pipeline: a Coordinator runs a Map step that fans out to four identical workers each processing a data chunk in parallel, then all four converge and combine into a single Reduce node producing the Merged Result](../../../public/images/agent/map-reduce.svg)

**Package:** [`agents/patterns/map-reduce-agent`](../../../agents/patterns/map-reduce-agent/README.md) (stub)

> **Note:**
> This is essentially [Parallelization](./parallelization)'s **Sectioning** variant, generalized: instead of a fixed set of subtasks written by the developer ahead of time, the number of parallel branches is determined **at runtime** by graph state. Same idea, now with unknown-in-advance fan-out.

## Core concepts

- **Older than any other pattern in this series** — MapReduce predates LLM agents by roughly two decades. The map/reduce shape is borrowed here, not invented for this use case.
- **Dynamic fan-out via `Send`** — the mechanism that makes this genuinely different from fixed Sectioning: a routing function returns a list of `Send` objects, one per unit of work, and the number of units comes from graph state at runtime, not from code written in advance.
- **Map, then reduce** — map: each `Send` spawns an independent worker invocation of the same node, with its own state; reduce: a downstream node collects and aggregates all worker outputs via a reducer.

## Implementing the pattern

#### 1. Define the graph state
- **Action**: Include a reducer field that collects worker outputs as they complete.
  ```python
  import operator
  from typing import Annotated
  from typing_extensions import TypedDict

  class OverallState(TypedDict):
      subjects: list[str]
      jokes: Annotated[list[str], operator.add]
  ```
- **Result**: `jokes` accumulates every worker's output via `operator.add`, regardless of how many workers actually run.

#### 2. Define the worker node
- **Action**: A node that processes one unit of work and returns output shaped to merge into the reducer.
  ```python
  class JokeState(TypedDict):
      subject: str

  def generate_joke(state: JokeState) -> dict:
      joke = llm.invoke(f"Write a short joke about {state['subject']}.").content
      return {"jokes": [joke]}
  ```
- **Result**: Each invocation of `generate_joke` runs against its own `subject`, independent of every other invocation.

#### 3. Define the mapping/routing function
- **Action**: Return a list of `Send` objects — one per item — determined by current state, not fixed in code.
  ```python
  from langgraph.types import Send

  def continue_to_jokes(state: OverallState):
      return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
  ```
- **Result**: The number of `Send` objects — and therefore the number of parallel workers — depends entirely on how many `subjects` are in state when this function runs.

#### 4. Define the reduce node
- **Action**: A node that runs once every `Send`-spawned worker has completed.
  ```python
  def combine_jokes(state: OverallState) -> dict:
      return {"summary": f"Generated {len(state['jokes'])} jokes."}
  ```
- **Result**: `combine_jokes` only fires after all dynamically-spawned workers have returned — LangGraph waits for every branch before proceeding.

#### 5. Assemble the graph
- **Action**: Wire the routing function into a conditional edge from the entry node, and route workers into the reduce node.
  ```python
  from langgraph.graph import StateGraph, START, END

  graph = StateGraph(OverallState)
  graph.add_node("generate_joke", generate_joke)
  graph.add_node("combine_jokes", combine_jokes)

  graph.add_conditional_edges(START, continue_to_jokes, ["generate_joke"])
  graph.add_edge("generate_joke", "combine_jokes")
  graph.add_edge("combine_jokes", END)

  app = graph.compile()
  ```
- **Result**: `app.invoke({"subjects": ["cats", "airports", "Mondays"]})` spawns exactly 3 parallel `generate_joke` workers — pass 10 subjects instead, and it spawns 10, with no change to the graph itself.

#### 6. Worked example
- **Action**: Run it with a variable-length input.
  ```python
  result = app.invoke({"subjects": ["cats", "airports", "Mondays", "printers"]})
  print(result["summary"])  # "Generated 4 jokes."
  ```
- **Result**: Fan-out tracked the input list's length automatically — this is what a *fixed* Sectioning implementation from [Parallelization](./parallelization) can't do without rewriting the graph for every possible subject count.

## Production notes

Per [Argha Dey Sarkar's production checklist](https://medium.com/@email2argha/%EF%B8%8Fdelegate-parallelize-synthesize-building-orchestrator-worker-workflows-with-langgraph-d01b767655c4) for Send-based workflows:

- *"Design deterministic reducers to merge worker outputs"* — the reduce step shouldn't depend on which order workers happen to finish in.
- *"Keep workers stateless and focused"* — each worker node should do one thing, with no dependency on other workers' state.
- *"Set limits on fan-out (avoid 1000s of workers at once)"* — an unbounded input list means an unbounded number of parallel LLM calls.
- *"Add retries & fallbacks for workers (timeouts, defaults)"* — a single stuck or failed worker shouldn't take down the whole reduce step.
- *"Monitor routing & synthesis with tools like MLflow"* — dynamic fan-out is harder to reason about statically than a fixed graph, so tracing matters more here, not less.

> **Warning:**
> Common mistakes to check for:
> - **`Send` fan-out doesn't always achieve full parallelism in practice** — [an open LangGraph issue](https://github.com/langchain-ai/langgraph/issues/3329) reports only 16 of 21 expected parallel branches actually running concurrently, with no error surfaced. Don't assume the API guarantees full concurrency — verify actual concurrent execution in your own environment, and check `max_concurrency` config if you need a hard bound.
> - **Unbounded fan-out from untrusted or large input** — a large `subjects` list spawns a proportionally large number of workers. Cap it.
> - **Non-deterministic reduce order** — if the reducer's result depends on which worker happens to finish first, results can vary run to run for the same input.
> - **Forgetting this only helps when work is genuinely independent** — same requirement as Parallelization's Sectioning: if workers secretly depend on each other, dynamic fan-out doesn't fix that.

## References

- [Dean & Ghemawat, "MapReduce: Simplified Data Processing on Large Clusters" (OSDI 2004)](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/)
- [LangGraph — Use the graph API: Map-Reduce and the Send API](https://docs.langchain.com/oss/python/langgraph/use-graph-api#map-reduce-and-the-send-api)
- [`Send` API reference](https://reference.langchain.com/python/langgraph/types/Send)
- [Argha Dey Sarkar — Orchestrator-Worker workflows with LangGraph (production checklist)](https://medium.com/@email2argha/%EF%B8%8Fdelegate-parallelize-synthesize-building-orchestrator-worker-workflows-with-langgraph-d01b767655c4)
- [langchain-ai/langgraph#3329 — Send fan-out not achieving full parallelism](https://github.com/langchain-ai/langgraph/issues/3329)
- [Parallelization](./parallelization)
