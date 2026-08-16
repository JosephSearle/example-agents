# Swarm / Handoffs

Per [OpenAI's cookbook](https://developers.openai.com/cookbook/examples/orchestrating_agents), a handoff is *"an agent (or routine) handing off an active conversation to another agent, much like when you get transfered to someone else on a phone call. Except in this case, the agents have complete knowledge of your prior conversation!"* One agent, mid-conversation, transfers control to a specialist — who picks up with full context rather than starting cold.

![Swarm / Handoffs topology: three peer agents (Agent A, Agent B, Agent C) arranged in a triangle with direct bidirectional handoff arrows between every pair and no central coordinator; control passes directly from peer to peer, with one agent active at a time](../../public/images/swarm-handoffs.svg)

> **Note:**
> Per the [current OpenAI Agents SDK orchestration guide](https://developers.openai.com/api/docs/guides/agents/orchestration): *"Start with one agent whenever you can. Add specialists only when they materially improve capability isolation, policy isolation, prompt clarity, or trace legibility."* Splitting too early just creates more prompts, more traces, and more approval surfaces — without necessarily making the workflow better.

> **Note:**
> **Handoffs vs. [Supervisor](./supervisor), the direct disambiguator** — straight from the OpenAI SDK docs: with handoffs, *"control moves to the specialist agent"* — it owns the final response. With an agent used as a callable tool (the Supervisor pattern), *"the manager keeps ownership of the reply"* — the main agent stays responsible for synthesizing the answer. If a specialist should take over the conversation entirely, use handoffs; if the main agent should stay in charge of what the user ultimately sees, use [Supervisor](./supervisor).

> **Warning:**
> **Tooling note.** [`openai/swarm`](https://github.com/openai/swarm), the original reference implementation, is deprecated — its own README states: *"Swarm is now replaced by the OpenAI Agents SDK, which is a production-ready evolution of Swarm."* Migrate to the Agents SDK for anything production-bound. "Swarm" remains the name most commonly used for this pattern in casual conversation (hence this file's name) — just don't build new work directly on that retired repo. If you're in the LangGraph ecosystem instead, [`langgraph-swarm-py`](https://github.com/langchain-ai/langgraph-swarm-py) implements the same shape there.

## Core concepts

- **The phone-transfer analogy** — full conversation history moves with the handoff. The receiving specialist isn't starting cold; it has everything said so far.
- **Self-directed handoff via transfer functions** — per the OpenAI cookbook, give each agent a `transfer_to_XXX` function as a tool per specialist it might hand off to: *"The model is smart enough to know to call this function when it makes sense to make a handoff!"* You don't write routing logic — you give the model the option and trust it to use it.
- **Persistent "active agent"** — per [`langgraph-swarm-py`](https://github.com/langchain-ai/langgraph-swarm-py): *"the system remembers which agent was last active, ensuring that on subsequent interactions, the conversation resumes with that agent"* — a follow-up message picks up with the same specialist rather than restarting at a default entry point.

## Implementing the pattern

#### 1. Define each specialist agent
- **Action**: A `create_agent()` instance per specialist domain.
  ```python
  from langchain.agents import create_agent

  billing_agent = create_agent(
      model="anthropic:claude-sonnet-4-6",
      tools=[lookup_invoice, issue_refund],
      system_prompt="You handle billing questions and refunds. Once handed a billing conversation, own it end to end.",
  )

  triage_agent = create_agent(
      model="anthropic:claude-sonnet-4-6",
      tools=[],  # transfer tools added next
      system_prompt="You triage incoming support requests and hand off to the right specialist.",
  )
  ```
- **Result**: Two agents — one that will own billing conversations entirely once handed one, and an entry point that starts every conversation.

#### 2. Give each agent transfer tools
- **Action**: A `create_handoff_tool()`-style function per specialist a given agent might transfer to.
  ```python
  from langgraph_swarm import create_handoff_tool

  transfer_to_billing = create_handoff_tool(
      agent_name="billing_agent",
      description="Transfer to the billing specialist for invoice or refund questions.",
  )

  triage_agent = create_agent(
      model="anthropic:claude-sonnet-4-6",
      tools=[transfer_to_billing],
      system_prompt="You triage incoming support requests and hand off to the right specialist.",
  )
  ```
- **Result**: `triage_agent` can now transfer control to `billing_agent` whenever the model judges it appropriate — no hardcoded "if billing keyword, route to billing" logic required.

#### 3. Assemble the swarm
- **Action**: Combine the agents into a swarm with a default entry point, compiled with a checkpointer so the active agent persists across turns.
  ```python
  from langgraph_swarm import create_swarm
  from langgraph.checkpoint.memory import InMemorySaver

  swarm = create_swarm(
      [triage_agent, billing_agent],
      default_active_agent="triage_agent",
  ).compile(checkpointer=InMemorySaver())
  ```
- **Result**: `swarm` starts every new conversation at `triage_agent`, but remembers whichever agent last took control for follow-up turns.

#### 4. Run a conversation and observe the handoff
- **Action**: Invoke the swarm across two turns in the same thread.
  ```python
  import uuid

  thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

  swarm.invoke(
      {"messages": [{"role": "user", "content": "I was charged twice for my subscription."}]},
      config=thread,
  )

  # Follow-up, same thread — resumes with billing_agent, not triage_agent
  swarm.invoke(
      {"messages": [{"role": "user", "content": "Can you refund the extra charge?"}]},
      config=thread,
  )
  ```
- **Result**: The first turn is triaged and handed off to `billing_agent`; the second turn, in the same thread, resumes directly with `billing_agent` — the swarm remembers who's currently in charge of the conversation.

#### 5. Worked example
- **Action**: The full picture — a support triage agent hands a billing question to a billing specialist, who owns the rest of that conversation end to end.
  ```python
  # Turn 1: triage_agent receives the request, recognizes it's billing-related,
  # calls transfer_to_billing — control moves to billing_agent.
  # Turn 2 onward: billing_agent is the active agent; the user is now
  # effectively talking directly to the billing specialist.
  ```
- **Result**: The user experiences one continuous conversation, but ownership of the response has moved entirely to the specialist best suited to it.

> **Warning:**
> Common mistakes to check for:
> - **Splitting into specialists before it's warranted** — per the SDK's own guidance, don't add specialists unless they materially improve capability isolation, policy isolation, prompt clarity, or trace legibility.
> - **Vague transfer conditions** — if the model can't tell when a `transfer_to_XXX` tool applies, handoffs will misfire, the same lesson tool descriptions carry throughout this series (see [Orchestrator-Workers](./orchestrator-workers), [ReAct Agent](./react-agent)).
> - **Confusing this with Supervisor** — if the main agent should stay responsible for the final answer rather than handing off ownership entirely, use [Supervisor](./supervisor) instead.
> - **Depending on the deprecated `openai/swarm` repo for new work** — build on the current OpenAI Agents SDK, or `langgraph-swarm-py` if you're in that ecosystem.

## References

- [OpenAI cookbook — Orchestrating Agents (origin of the handoff concept)](https://developers.openai.com/cookbook/examples/orchestrating_agents)
- [`openai/swarm`](https://github.com/openai/swarm) (deprecated — historical reference only)
- [OpenAI Agents SDK — Orchestration guide (current)](https://developers.openai.com/api/docs/guides/agents/orchestration)
- [`langgraph-swarm-py`](https://github.com/langchain-ai/langgraph-swarm-py) (LangGraph-ecosystem equivalent)
- [Supervisor](./supervisor)
