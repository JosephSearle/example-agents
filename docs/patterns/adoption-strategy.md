# Workflow & Agent Adoption Strategy

Per [Anthropic's own guide](https://www.anthropic.com/engineering/building-effective-agents), the mission statement this whole series is built around: *"the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns."* This series covers ten patterns across three tiers. This doc isn't another pattern — it's how to decide which one(s) you actually need, and in what order, before writing any code.

> **Note:**
> There's no single paper behind this doc — this is a synthesis of the ten patterns already documented in this series. The References section below is background reading on the broader agent-architecture landscape, not the source this strategy was derived from.

## The three tiers

Per Anthropic's own progression: start with a single, well-optimized LLM call; move to a workflow only when that proves insufficient; reach for an agent only for open-ended tasks needing autonomous decision-making over steps that can't be predicted in advance. Complexity should only be added "when it demonstrably improves outcomes."

| Tier | What it means | Patterns in this series |
|---|---|---|
| **Single LLM call** | Not a pattern in this series — the actual starting point. Optimize with retrieval and in-context examples before reaching for anything below. | — |
| **Workflows** | Fixed code paths — the sequence (or fan-out shape) is decided by the developer, not the model | [Prompt Chaining](./prompt-chaining), [Routing](./routing), [Parallelization](./parallelization), [Map-Reduce](./map-reduce), [Orchestrator-Workers](./orchestrator-workers), [Evaluator-Optimizer](./evaluator-optimizer) |
| **Single agent** | The LLM dynamically directs its own process — reasoning, acting, and stopping on its own | [ReAct Agent](./react-agent) |
| **Multi-agent** | Multiple agents, with dynamic coordination between them | [Supervisor](./supervisor), [Swarm / Handoffs](./swarm-handoffs), [Network / Mesh](./network-mesh) |

[Map-Reduce](./map-reduce) sits in the workflow tier as a dynamic-fan-out generalization of [Parallelization](./parallelization)'s Sectioning — still a fixed code path, just one where the worker count is decided by state at runtime rather than by the developer ahead of time.

## Deciding what you need

#### 1. Start with a single LLM call
- Always the baseline. Don't reach for any pattern in this series until a well-optimized single call — good retrieval, good in-context examples, a clear prompt — has actually proven insufficient.

#### 2. Is the task decomposable into fixed, predictable steps?
- If yes, you want a workflow — and the specific choice depends on the shape of the decomposition:
  - Sequential fixed steps, each building on the last → [Prompt Chaining](./prompt-chaining).
  - Classify the input, then dispatch to a specialized path → [Routing](./routing).
  - Independent parallel work with a known, fixed shape → [Parallelization](./parallelization).
  - Independent parallel work whose *count* isn't known until runtime → [Map-Reduce](./map-reduce).
  - Subtasks that can't be predicted in advance and must be decided dynamically → [Orchestrator-Workers](./orchestrator-workers).
  - Iterative refinement against clear evaluation criteria → [Evaluator-Optimizer](./evaluator-optimizer).

#### 3. Does the task need open-ended, autonomous decision-making, but one model with tools is still enough?
- If the steps genuinely can't be predicted in advance — the model needs to reason, act, and decide what to do next on its own, over many steps — but a single agent can hold all the context and tools it needs → [ReAct Agent](./react-agent).

#### 4. Has a single well-tooled agent actually plateaued?
- Per [Supervisor](./supervisor)'s cited rule of thumb: stay with a single agent if it's hitting ≥85% on your eval and your tasks are reasonably homogeneous. Only move to multi-agent once you have demonstrably distinct task domains and a single agent's accuracy has plateaued despite genuine prompt/tool iteration — not preemptively.

#### 5. Within multi-agent — which topology?
- Should one agent stay responsible for the final answer, delegating bounded sub-tasks to specialists that never talk to the user directly? → [Supervisor](./supervisor).
- Should a specialist fully take over the conversation once handed it? → [Swarm / Handoffs](./swarm-handoffs).
- Is there genuinely no hierarchy or fixed order at all? → [Network / Mesh](./network-mesh) — reached for last, given its comparatively weak tooling support relative to the other two multi-agent shapes.

## Core guidance

Anthropic's mission statement is the throughline for every step above: adopt in order of the failure you've actually diagnosed, not in order of sophistication. For the specific jump from single-agent to multi-agent, the clearest real production account of when it actually paid off is [Anthropic's own multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) (already cited in [Orchestrator-Workers](./orchestrator-workers)) — a lead agent delegating to subagents only once the task genuinely benefited from parallel, independent exploration of different aspects of a query.

> **Warning:**
> Common mistakes to check for:
> - **Jumping to an agent or multi-agent tier before confirming a single call or workflow genuinely can't do the job** — every tier above the baseline adds real latency, cost, and unpredictability.
> - **Stacking multiple workflow patterns before confirming each pulls its weight individually** — the same warning already given in [Parallelization](./parallelization) and [Evaluator-Optimizer](./evaluator-optimizer) applies across the whole series, not just within one doc.
> - **Treating "agent" as inherently better than "workflow"** — cost, latency, and predictability tradeoffs are real and already documented per-pattern throughout this series; more autonomy is not automatically a better outcome.
> - **Defaulting to Network/Mesh's flexibility** when a hierarchy ([Supervisor](./supervisor)) or a handoff chain ([Swarm / Handoffs](./swarm-handoffs)) would actually fit — both are better-supported and easier to reason about.

## References

Background reading on the broader agent-architecture landscape — not the source this strategy is derived from; the adoption logic above synthesizes the patterns and papers already cited throughout this series.

- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Masterman et al., "The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling" (arXiv:2404.11584, 2024)](https://arxiv.org/abs/2404.11584)
- ["Designing LLM-based Multi-Agent Systems for Software Engineering Tasks" (arXiv:2511.08475, 2025)](https://arxiv.org/pdf/2511.08475) — see also, for a formal design-pattern catalog covering similar ground to the multi-agent tier above
- [Prompt Chaining](./prompt-chaining)
- [Routing](./routing)
- [Parallelization](./parallelization)
- [Orchestrator-Workers](./orchestrator-workers)
- [Evaluator-Optimizer](./evaluator-optimizer)
- [ReAct Agent](./react-agent)
- [Supervisor](./supervisor)
- [Swarm / Handoffs](./swarm-handoffs)
- [Map-Reduce](./map-reduce)
- [Network / Mesh](./network-mesh)
