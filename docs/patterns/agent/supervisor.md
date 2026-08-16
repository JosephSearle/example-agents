# Supervisor

A central supervisor agent delegates to specialized sub-agents, each scoped to its own domain and tools, while the supervisor itself never touches low-level tools directly — it only calls sub-agents as tools and coordinates the overall workflow.

![Supervisor topology: a central Supervisor coordinator connected by bidirectional arrows to three workers (Researcher, Analyst, Writer), with no direct links between the workers; all control flow returns through the Supervisor before reaching Output](../../../public/images/agent/supervisor.svg)

> **Note:**
> Per [LangChain's current guidance](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant): *"Use the supervisor pattern when you have multiple distinct domains (calendar, email, CRM, database), each domain has multiple tools or complex logic, you want centralized workflow control, and sub-agents don't need to converse directly with users."* For simpler cases with just a few tools, use a single agent instead. And critically: *"When agents need to have conversations with users, use handoffs instead."* See [Swarm / Handoffs](./swarm-handoffs) — the user ends up talking directly to a specialist rather than always through one orchestrator.

> **Warning:**
> **Tooling note.** The standalone [`langgraph-supervisor-py`](https://github.com/langchain-ai/langgraph-supervisor-py) library's own README now says: *"We now recommend using the supervisor pattern directly via tools rather than this library for most use cases. The tool-calling approach gives you more control over context engineering and is the recommended pattern in the LangChain multi-agent guide."* This doc's walkthrough uses that manual tool-calling approach. If you're maintaining existing code built on `create_supervisor()` from that library, it still works — just don't reach for it on new work.

## Core concepts

- **Sub-agents as tools, not peers** — each sub-agent is a full agent in its own right, but from the supervisor's point of view it's just another tool call: the supervisor sees the sub-agent's final output, not its internal reasoning.
- **Centralized control, no direct user contact for sub-agents** — the supervisor is the only thing the user talks to. Sub-agents never converse with the user directly — that's the boundary against handoffs.
- **Supervisor vs. handoffs, the direct disambiguator** — does the user always talk to one orchestrator that delegates internally (supervisor), or get handed off to a specialist they now talk to directly (handoffs)?
- **Real cost/latency tradeoff** — one team's production benchmark ([CallSphere, 2026](https://callsphere.ai/blog/langgraph-supervisor-multi-agent-orchestration-2026), not peer-reviewed): supervisor adds *"~1.5–2x wall-clock vs. a single agent"* and *"2–3x [cost]"* in their measurements. Their rule of thumb: *"if a single agent with all the tools is hitting ≥85% on your eval and your tasks are reasonably homogeneous, stay single."* Move to supervisor *"when you have demonstrably distinct task types and a single agent's accuracy plateaus despite prompt iteration."*

## Implementing the pattern

#### 1. Build each sub-agent
- **Action**: A `create_agent()` instance scoped to one domain's tools.
  ```python
  from langchain.agents import create_agent

  calendar_agent = create_agent(
      model="anthropic:claude-sonnet-4-6",
      tools=[list_events, create_event, cancel_event],
      system_prompt="You manage the user's calendar. Handle scheduling requests precisely.",
  )

  email_agent = create_agent(
      model="anthropic:claude-sonnet-4-6",
      tools=[search_inbox, send_email, draft_reply],
      system_prompt="You manage the user's email. Draft and send messages on request.",
  )
  ```
- **Result**: Two fully independent agents, each an expert in exactly one domain's tools.

#### 2. Wrap each sub-agent as a tool
- **Action**: A thin `@tool`-decorated function invokes the sub-agent and returns its final output.
  ```python
  from langchain_core.tools import tool

  @tool
  def delegate_to_calendar(request: str) -> str:
      """Delegate a calendar-related request to the calendar sub-agent."""
      result = calendar_agent.invoke({"messages": [{"role": "user", "content": request}]})
      return result["messages"][-1].content

  @tool
  def delegate_to_email(request: str) -> str:
      """Delegate an email-related request to the email sub-agent."""
      result = email_agent.invoke({"messages": [{"role": "user", "content": request}]})
      return result["messages"][-1].content
  ```
- **Result**: Two tools that look, from the supervisor's perspective, exactly like any other tool call — the sub-agent's internal reasoning is hidden.

#### 3. Build the supervisor
- **Action**: Another `create_agent()` instance whose only tools are the wrapped sub-agents.
  ```python
  supervisor = create_agent(
      model="anthropic:claude-sonnet-4-6",
      tools=[delegate_to_calendar, delegate_to_email],
      system_prompt=(
          "You are a personal assistant. Delegate calendar requests to "
          "delegate_to_calendar and email requests to delegate_to_email. "
          "Never handle domain-specific requests yourself."
      ),
  )
  ```
- **Result**: `supervisor` has no direct access to low-level tools like `send_email` or `create_event` — only to the sub-agents that own them.

#### 4. Invoke the supervisor
- **Action**: The user only ever talks to the supervisor; it decides which sub-agent(s) to call.
  ```python
  import uuid

  result = supervisor.invoke(
      {"messages": [{"role": "user", "content": "Move my 3pm meeting to 4pm."}]},
      config={"configurable": {"thread_id": str(uuid.uuid4())}},
  )
  ```
- **Result**: The supervisor routes the request to `delegate_to_calendar` without the user ever interacting with the calendar sub-agent directly.

#### 5. Worked example
- **Action**: A request that spans two domains in one turn.
  ```python
  thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

  supervisor.invoke(
      {
          "messages": [
              {
                  "role": "user",
                  "content": "Move my 3pm meeting to 4pm, and email the attendees to let them know.",
              }
          ]
      },
      config=thread,
  )
  ```
- **Result**: The supervisor calls `delegate_to_calendar` for the reschedule and `delegate_to_email` to notify attendees, synthesizing both into one response — without either sub-agent needing to know the other exists.

> **Warning:**
> Common mistakes to check for:
> - **Reaching for supervisor before a single agent has actually plateaued** — per CallSphere's rule of thumb, don't add multi-agent complexity until a single well-tooled agent has demonstrably hit its ceiling on your eval.
> - **Conflating supervisor with handoffs** — if sub-agents need to talk to users directly, this is the wrong pattern.
> - **Underestimating cost** — budget for the ~1.5–2x latency / 2–3x cost multiplier reported by one team's benchmark before committing to this architecture; treat it as directional, not a universal constant.
> - **Depending on `langgraph-supervisor-py` for new work** — the library itself now points new users toward the manual tool-calling approach used above.

## References

- [LangChain — Supervisor / subagents (current docs)](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant)
- [`langgraph-supervisor-py`](https://github.com/langchain-ai/langgraph-supervisor-py) (soft-deprecated in favor of the manual tool-calling approach)
- [LangGraph Supervisor Pattern: Orchestrating Multi-Agent Teams (CallSphere, 2026)](https://callsphere.ai/blog/langgraph-supervisor-multi-agent-orchestration-2026) — one team's production benchmark, not peer-reviewed
- [ReAct Agent](./react-agent)
