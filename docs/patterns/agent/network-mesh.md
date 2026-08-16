# Network / Mesh

Agents that can all talk to each other, many-to-many, with no fixed hierarchy — any agent can decide which agent to call next.

![Network / Mesh topology: five agents (Planner, Executor, Reviewer, Researcher, Validator) scattered in a loose non-symmetric layout with non-uniform connections including a route-back edge; any agent can route to any other, and the traversal path is decided at runtime rather than by a fixed graph shape](../../../public/images/agent/network-mesh.svg)

**Package:** [`agents/patterns/network-mesh-agent`](../../../agents/patterns/network-mesh-agent/README.md) (stub)

> **Warning:**
> **Honest framing up front.** Of the four multi-agent topologies in this series ([Supervisor](./supervisor), [Swarm / Handoffs](./swarm-handoffs), [Map-Reduce](./map-reduce), and this one), Network/Mesh is the only one without a dedicated, actively-maintained first-party doc page or standalone library. It shows up mostly in archived tutorials and third-party content. That's a real signal about the pattern's actual industry standing, not a gap in research — and it's itself a reason to reach for this pattern last, only once Supervisor's hierarchy and Swarm's peer-handoff both genuinely don't fit.

## Origin

Network/Mesh isn't a LangGraph-original idea. Its conceptual origin is [AutoGen (Wu et al., COLM 2024)](https://arxiv.org/abs/2308.08155): *"AutoGen is an open-source framework that allows developers to build LLM applications via multiple agents that can converse with each other to accomplish tasks... developers can flexibly define agent interaction behaviors."* AutoGen popularized the idea of multiple conversable agents freely messaging each other to solve a task collaboratively, without a fixed hierarchy — exactly the "any agent can talk to any other agent" shape this doc describes.

> **Note:**
> Per the [LangChain OpenTutorial](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/17-langgraph/02-structures/08-langgraph-multi-agent-structures-01), the cleanest plain-English definition available: *"In this architecture, agents are defined as graph nodes. Each agent can communicate with all other agents (many-to-many connections) and can decide which agent to call next."* And: *"This architecture is suitable for problems where there is no clear hierarchy of agents or specific order in which agents must be called."* This is the direct opposite of [Supervisor](./supervisor)'s fixed hub-and-spoke control, where the supervisor is the only thing that ever decides what happens next.

> **Warning:**
> **Tooling note.** [LangGraph's original multi-agent network/collaboration tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/) now redirects with no content — the same archival pattern already seen for the CRAG and Self-RAG notebooks elsewhere in this series. Unlike those, though, there's no clear current replacement page specifically for this topology. Cite the original tutorial as the canonical conceptual example, not as a currently-maintained reference.

## Core concepts

- **Many-to-many, no central coordinator** — every agent is a peer; any agent can call any other agent.
- **Each agent decides who's next** — routing isn't fixed to one supervisor or one handoff chain; it's distributed across however many agents are in the mesh.
- **Best fit: genuinely no hierarchy or fixed order** — if there actually is a natural hierarchy (one agent should own final answers) or a natural handoff chain (one specialist per conversation), [Supervisor](./supervisor) or [Swarm / Handoffs](./swarm-handoffs) are both better-supported, better-documented choices.

## Implementing the pattern

#### 1. Define shared state
- **Action**: A single state object every agent reads from and writes to.
  ```python
  from typing import Annotated
  from typing_extensions import TypedDict
  import operator

  class MeshState(TypedDict):
      task: str
      messages: Annotated[list[str], operator.add]
      next_agent: str
  ```
- **Result**: `messages` accumulates contributions from every agent that acts, and `next_agent` carries each agent's routing decision.

#### 2. Define each agent as a node
- **Action**: Each agent's node function does its work and appends its contribution to shared state.
  ```python
  def researcher(state: MeshState) -> dict:
      finding = llm.invoke(f"Research: {state['task']}").content
      return {"messages": [f"[researcher] {finding}"]}

  def critic(state: MeshState) -> dict:
      critique = llm.invoke(
          f"Critique this research for gaps:\n\n{state['messages'][-1]}"
      ).content
      return {"messages": [f"[critic] {critique}"]}

  def writer(state: MeshState) -> dict:
      draft = llm.invoke(
          f"Write a final answer using:\n\n" + "\n\n".join(state["messages"])
      ).content
      return {"messages": [f"[writer] {draft}"]}
  ```
- **Result**: Three peer agents, none of which is designated as the entry point or the final authority by default.

#### 3. Each agent's routing function decides the next agent
- **Action**: A routing function per agent, deciding who should act next based on current state.
  ```python
  from typing import Literal

  def route_after_researcher(state: MeshState) -> Literal["critic", "writer"]:
      decision = llm.invoke(
          f"Given this research, should a critic review it first, or is it "
          f"ready to write up? Reply 'critic' or 'writer'.\n\n{state['messages'][-1]}"
      ).content.strip().lower()
      return "critic" if "critic" in decision else "writer"
  ```
- **Result**: The next agent isn't fixed in the graph's structure — it's decided per invocation, based on what the current agent's output actually looks like.

#### 4. Assemble the graph
- **Action**: Wire every agent node to every peer it might route to — no single entry-only hub.
  ```python
  from langgraph.graph import StateGraph, START, END

  graph = StateGraph(MeshState)
  graph.add_node("researcher", researcher)
  graph.add_node("critic", critic)
  graph.add_node("writer", writer)

  graph.add_edge(START, "researcher")
  graph.add_conditional_edges("researcher", route_after_researcher, ["critic", "writer"])
  graph.add_edge("critic", "writer")
  graph.add_edge("writer", END)

  app = graph.compile()
  ```
- **Result**: `app` starts at `researcher`, but which path it takes afterward — straight to `writer`, or via `critic` first — depends on the routing decision made at runtime, not on a fixed sequence.

#### 5. Worked example
- **Action**: Run the mesh on a task and observe that the path isn't predetermined.
  ```python
  result = app.invoke({"task": "Summarize recent trends in vector database indexing.", "messages": [], "next_agent": ""})
  print(result["messages"])
  ```
- **Result**: For a well-covered topic, the researcher's output might go straight to the writer; for a thin or ambiguous topic, it routes through the critic first — the same graph handles both without being told in advance which path a given task needs.

> **Warning:**
> **Practical caveats.** Per the [Augment Code multi-agent orchestration guide](https://www.augmentcode.com/guides/multi-agent-orchestration-architecture-guide): mesh communication pathways scale as *"O(N²), with N being the agent count"* — every additional peer multiplies the number of possible communication paths, not just adds to it. And critically: *"Without a global state owner, parallel agents can produce overlapping changes from partial context, leading to merge conflicts and semantic contradictions."* Every agent reading and writing the same state object means you need a deliberate merge/ownership strategy per field — not an implicit one. In practice, this also makes failures harder to debug than in a single-agent system: a bad output several steps earlier, now sitting in shared state, doesn't leave behind a clean call stack pointing back to its source the way a single agent's failure would.
>
> Common mistakes to check for:
> - **Reaching for mesh before confirming Supervisor/Swarm genuinely don't fit** — both are better-supported and easier to reason about; only use mesh when neither's structure (a hierarchy, or a handoff chain) actually matches the problem.
> - **Too many agents in the mesh** — communication complexity compounds as O(N²); a handful of peers is very different from a dozen.
> - **No clear merge strategy for shared state** — decide, per field, who's allowed to write it and how conflicting writes are resolved, rather than leaving it implicit.

## References

- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation (Wu et al., COLM 2024, arXiv:2308.08155)](https://arxiv.org/abs/2308.08155)
- [LangChain OpenTutorial — Multi-Agent Structures](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/17-langgraph/02-structures/08-langgraph-multi-agent-structures-01)
- [LangGraph's original multi-agent network/collaboration tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/) (archived — no current first-party replacement)
- [Augment Code — Multi-Agent Orchestration Architecture Guide](https://www.augmentcode.com/guides/multi-agent-orchestration-architecture-guide)
- [Supervisor](./supervisor)
- [Swarm / Handoffs](./swarm-handoffs)
