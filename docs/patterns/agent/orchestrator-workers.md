# Orchestrator-Workers

Have you ever needed multiple perspectives on the same task, but couldn't predict in advance which perspectives would be most valuable? That's the problem orchestrator-workers solves: a central LLM (the orchestrator) dynamically breaks down a task into subtasks, delegates each to a worker LLM, and synthesizes their results into a final answer.

![Orchestrator-Workers workflow: a Task goes to a central Orchestrator that dynamically decides how many workers to spawn, delegates to a variable set of workers (Worker 1, Worker 2, and further ghosted workers), which feed back into Orchestrator Synthesize and then Output](../../../public/images/agent/orchestrator-workers.svg)

> **Note:**
> This is a boundary case in Anthropic's own taxonomy. Orchestrate → workers → synthesize is a fixed high-level shape, so it's still framed as a workflow — but it's genuinely different from [Prompt Chaining](./prompt-chaining), [Routing](./routing), and [Parallelization](./parallelization): the *number and nature* of subtasks is decided by the orchestrator LLM at runtime, not fixed in code. Per [Anthropic's own framing](https://www.anthropic.com/engineering/building-effective-agents): *"the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input."*

## Core concepts

- **Dynamic vs. fixed decomposition** — in [Parallelization](./parallelization)'s Sectioning tab, the developer writes each subtask ahead of time. Here, the orchestrator LLM decides what the subtasks are, and how many there are, based on the specific input it's given.
- **Three-stage shape** — orchestrate (analyze the task and decompose it), workers (execute each subtask, typically in parallel), synthesize (combine worker outputs into one coherent result).
- **Cost has three terms, and it's variable** — the orchestrator's planning call, N worker calls, and a synthesis call. Because the orchestrator decides N based on the input, cost varies per input in a way none of the fixed-shape patterns before it do.
- **Model-tiering as a cost lever** — a stronger model (e.g. Opus) for the orchestrator, since decomposition and planning benefit from stronger reasoning, and a cheaper model (e.g. Haiku) for workers executing well-scoped subtasks.

## Implementing the pattern

> **Note:**
> The [Anthropic cookbook's reference implementation](https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/orchestrator_workers.ipynb) uses a `FlexibleOrchestrator` class with XML-tagged structured output (`<analysis>`, `<tasks>`, `<task>`) parsed via an `extract_xml` helper. The walkthrough below uses Pydantic structured output instead, for consistency with the rest of this series (e.g. [Routing](./routing)) — it's a stylistic translation of the same idea, not a line-for-line port of the notebook's code.

#### 1. Define the orchestrator's output shape
- **Action**: Declare the structure the orchestrator must decompose a task into — a list of subtasks, each with a description of what that subtask needs to do.
  ```python
  from pydantic import BaseModel, Field

  class Subtask(BaseModel):
      description: str = Field(description="What this subtask needs to accomplish.")

  class TaskBreakdown(BaseModel):
      analysis: str = Field(description="Why this decomposition fits the input.")
      subtasks: list[Subtask]
  ```
- **Result**: `TaskBreakdown` is what the orchestrator must produce — not a fixed count of subtasks, but however many the specific input actually calls for.

#### 2. Orchestrator analyzes and decomposes the task
- **Action**: One LLM call, using the stronger model, produces a dynamic, input-dependent list of subtasks.
  ```python
  orchestrator = strong_model.with_structured_output(TaskBreakdown)

  def orchestrate(task: str) -> TaskBreakdown:
      return orchestrator.invoke(
          f"Break down this task into the subtasks needed to complete it. "
          f"The number and nature of subtasks should fit this specific task, "
          f"not follow a fixed template.\n\nTask: {task}"
      )
  ```
- **Result**: `orchestrate` returns a breakdown genuinely shaped by the input — a simple task might yield two subtasks, a complex one might yield ten.

#### 3. Workers execute the subtasks in parallel
- **Action**: Run a worker call for every subtask the orchestrator produced — using the cheaper model, since each subtask is now well-scoped.
  ```python
  import asyncio

  async def run_worker(subtask: Subtask) -> str:
      result = await weak_model.ainvoke(subtask.description)
      return result.content

  async def run_workers(breakdown: TaskBreakdown) -> list[str]:
      return await asyncio.gather(*[run_worker(s) for s in breakdown.subtasks])
  ```
- **Result**: `run_workers` executes however many subtasks the orchestrator decided on — the list itself isn't hardcoded, unlike Parallelization's Sectioning.

#### 4. Synthesize the results
- **Action**: One final LLM call combines every worker's output into a coherent result.
  ```python
  def synthesize(task: str, worker_results: list[str]) -> str:
      combined = "\n\n".join(f"Subtask result {i+1}:\n{r}" for i, r in enumerate(worker_results))
      return strong_model.invoke(
          f"Combine these subtask results into one coherent response to the "
          f"original task.\n\nOriginal task: {task}\n\n{combined}"
      ).content
  ```
- **Result**: A single, synthesized answer — not a raw dump of N independent worker outputs.

#### 5. Worked example
- **Action**: A multi-file code change — the orchestrator decides which files need touching and what kind of change each needs; workers each handle one file; synthesis produces a summary of the full change set.
  ```python
  async def handle_code_change(task: str) -> str:
      breakdown = orchestrate(task)
      worker_results = await run_workers(breakdown)
      return synthesize(task, worker_results)

  # e.g. task = "Add rate limiting to the API — touch whatever files that requires."
  ```
- **Result**: For this task, the orchestrator might decide it needs to touch a middleware file, a config file, and a test file — a different task might need a completely different set of files, decided at runtime rather than predicted in advance.

## Real-world case study: Anthropic's multi-agent research system

Anthropic's own [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) runs this exact pattern in production. A lead agent (the orchestrator) analyzes an incoming query, develops a research strategy, and spawns subagents (workers) to explore different aspects of the query simultaneously. Each subagent acts as an intelligent filter — iteratively using search tools to gather information — before returning its findings to the lead agent, which compiles them into a final answer.

The system's own postmortem gives a directly useful lesson for this pattern: *"Without detailed task descriptions, agents duplicate work, leave gaps, or fail to find necessary information."* In one documented failure, a subagent explored the 2021 automotive chip crisis while two other subagents duplicated each other investigating current 2025 supply chains — no effective division of labor. A well-scoped subagent task description needs four things: an explicit objective, an output format, tool guidance, and clear task boundaries.

> **Warning:**
> Common mistakes to check for:
> - **Vague worker/subagent task descriptions** — per the case study above, this is what actually causes duplicated work, coverage gaps, or missed information in practice. Every subtask description needs an explicit objective, output format, tool guidance, and clear boundaries.
> - **Non-deterministic cost** — this pattern doesn't have a fixed cost the way the earlier patterns in this series do. Budget for the fact that N varies per input, and can spike on complex inputs.
> - **Reaching for this when subtasks are actually predictable** — if you already know the subtasks ahead of time, that's [Parallelization](./parallelization)'s Sectioning, which is cheaper and more predictable. Don't default to dynamic decomposition when a fixed one would do.
> - **Skipping model-tiering** — paying premium-model cost for every worker call regardless of how simple the subtask is erodes the cost benefit this pattern is supposed to have relative to running everything on the strongest model.

## Framework equivalents

This shape appears under different names across other frameworks — useful context if you're comparing tooling rather than building from scratch: LangGraph's **supervisor pattern** (the orchestrator-worker shape with a typed message contract between supervisor and workers), CrewAI's **hierarchical process** (a manager agent plans and delegates to workers), and AutoGen **Teams** (a similar orchestrator-and-worker abstraction). *(Framework-equivalents note sourced from [buildingeffectiveagents.com](https://buildingeffectiveagents.com/patterns/orchestrator-worker/), a third-party reference, not Anthropic directly.)*

## References

- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Anthropic — Multi-agent research system (case study)](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Anthropic cookbook — `orchestrator_workers.ipynb`](https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/orchestrator_workers.ipynb)
- [buildingeffectiveagents.com — Orchestrator-Worker (secondary/third-party)](https://buildingeffectiveagents.com/patterns/orchestrator-worker/)
- [Parallelization](./parallelization)
- [Routing](./routing)
