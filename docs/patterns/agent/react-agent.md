# ReAct Agent

ReAct interleaves reasoning ("thoughts") with actions (tool calls) in a loop: the model reasons about what to do next, acts, observes the result, and reasons again — continuing until the task is done. Unlike every pattern before it in this series, the loop's length and shape aren't fixed in code. The model decides.

![ReAct Agent reason and act loop: a User Task enters the Thought node; the agent cycles clockwise through Thought, Action, and Observation repeatedly, and once it decides the task is done it exits from Thought to the Final Answer](../../../public/images/agent/react-agent.svg)

> **Note:**
> Every prior doc in this series has been a **workflow** — chaining, routing, parallelization, orchestrator-workers, and evaluator-optimizer all follow a code path fixed in advance. This is the first genuine **agent**: the LLM dynamically directs its own process, deciding when to reason, when to act, and when to stop.

> **Warning:**
> **Tooling note.** `create_react_agent` (`langgraph.prebuilt`) is the historical function — it's literally had "ReAct" in its name for years and is what most existing tutorials, blog posts, and cookbooks reference. It now carries an explicit deprecation notice, superseded by `create_agent` (`langchain.agents`), which is what this doc uses. The migration isn't yet a strict superset, though — an active LangChain community forum thread (October 2025) flagged a real functional gap: loss of the ability to rewrite message history as a function of state. If you depend on that specific capability, evaluate carefully before migrating off `create_react_agent`.

## Core concepts

- **Thought → Action → Observation, looped** — the paper's core mechanism: reason about the next step, take an action (usually a tool call), observe the result, repeat.
- **Why interleaving reasoning and acting helps** — reasoning traces let the model track and update its own plan and handle exceptions; actions let it pull in real information rather than relying on parametric memory alone. Each strengthens the other.
- **The harness** — per LangChain's framing, an agent is the model plus a *harness*: system prompt, tools, and middleware together, giving the model the right context at the right time. Tool descriptions specifically double as the model's only guide to what actions are available and when to use them.

## Implementing the pattern

#### 1. Define tools
- **Action**: Plain Python callables with clear docstrings — the model reads these to decide when and how to act.
  ```python
  def check_weather(location: str) -> str:
      """Return the current weather forecast for the given location."""
      return weather_client.get_forecast(location)

  def search_docs(query: str) -> str:
      """Search internal documentation and return the most relevant excerpt."""
      return doc_search_client.search(query)
  ```
- **Result**: Two tools the agent can choose to call, each with a docstring precise enough for the model to know when it applies.

#### 2. Create the agent
- **Action**: `create_agent` wires the model, tools, and system prompt into a runnable agent.
  ```python
  from langchain.agents import create_agent

  agent = create_agent(
      model="anthropic:claude-sonnet-4-6",
      tools=[check_weather, search_docs],
      system_prompt="You are a helpful assistant. Use tools when you need current information.",
  )
  ```
- **Result**: `agent` is ready to run the Thought/Action/Observation loop internally — nothing about how many tool calls it makes, or in what order, is decided in this code.

#### 3. Add a checkpointer for multi-turn persistence
- **Action**: A checkpointer preserves conversation history across turns.
  ```python
  from langgraph.checkpoint.memory import InMemorySaver

  agent = create_agent(
      model="anthropic:claude-sonnet-4-6",
      tools=[check_weather, search_docs],
      system_prompt="You are a helpful assistant. Use tools when you need current information.",
      checkpointer=InMemorySaver(),
  )
  ```
- **Result**: The agent remembers prior turns within the same `thread_id` — required for any multi-turn conversation. (`InMemorySaver` is fine for local development; swap in a persistent checkpointer for production.)

#### 4. Invoke the agent
- **Action**: Call `.invoke()` with a message and a `thread_id` to scope conversation state.
  ```python
  import uuid

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
      config={"configurable": {"thread_id": str(uuid.uuid4())}},
  )
  ```
- **Result**: `result` contains the full message trace — including any tool calls the agent chose to make — ending in a final response.

#### 5. Worked example
- **Action**: A question that requires a tool call and a follow-up reasoning step, showing the loop happen without being manually orchestrated.
  ```python
  thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

  agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in Tokyo, and is that good for sightseeing?"}]},
      config=thread,
  )
  ```
- **Result**: The agent reasons that it needs current weather data, calls `check_weather("Tokyo")`, observes the result, and reasons again to judge whether that forecast is good for sightseeing — all inside a single `.invoke()` call, with no explicit "call the tool, then reason" code written by you.

## Historical reference

Most existing tutorials still use the older shape:

```python
from langgraph.prebuilt import create_react_agent

graph = create_react_agent(
    "anthropic:claude-sonnet-4-6",
    tools=[check_weather],
    prompt="You are a helpful assistant",
)
```

Recognize this pattern when you see it in older material — see the tooling callout above for why this doc leads with `create_agent` instead. A minimal, official starting template for this pattern is also available at [`langchain-ai/react-agent`](https://github.com/langchain-ai/react-agent).

> **Warning:**
> Common mistakes to check for:
> - **Unbounded loops** — an agent can call tools indefinitely without a recursion or iteration cap. Set one.
> - **Tool description quality drives decision quality** — vague tool docstrings produce bad action choices, the same lesson [Orchestrator-Workers](./orchestrator-workers) already established for subtask descriptions.
> - **Fully dynamic cost/latency** — unlike every fixed-shape pattern earlier in this series, the number of loop iterations isn't predictable ahead of time.
> - **Reaching for an agent when a workflow would do** — if the task's steps are actually predictable, a workflow pattern from earlier in this series is cheaper and more predictable than giving up control to an agent loop.

## References

- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., ICLR 2023, arXiv:2210.03629)](https://arxiv.org/abs/2210.03629)
- [LangChain — Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [`create_react_agent` API reference (historical, deprecated)](https://reference.langchain.com/python/langgraph.prebuilt/chat_agent_executor/create_react_agent)
- [`langchain-ai/react-agent` — official template](https://github.com/langchain-ai/react-agent)
- [Evaluator-Optimizer](./evaluator-optimizer)
- [Prompt Chaining](./prompt-chaining)
