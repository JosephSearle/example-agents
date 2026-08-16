# Prompt Chaining

Prompt chaining decomposes one complex task into a sequence of simpler LLM calls, where each step's output feeds directly into the next — rather than asking a single prompt to do everything at once. It trades latency (multiple LLM round-trips) for accuracy: each individual call is a smaller, more focused task, and is less likely to fail at doing too much at once.

![Prompt Chaining pipeline for contract analysis: Input, Step 1 Analyze, Step 2 Extract Terms, Step 3 Check Compliance, Step 4 Summarize, Output — with a small Check validation gate on each connecting arrow between steps](../../../public/images/agent/prompt-chaining.svg)

> **Note:**
> This is a **workflow**, not an agent, per [Anthropic's framing](https://www.anthropic.com/engineering/building-effective-agents): the sequence of steps is fixed in code ahead of time — the LLM doesn't decide what step comes next or whether to loop back. It's best suited to tasks that decompose cleanly into fixed subtasks. If your task needs dynamic branching, or the LLM deciding its own next step, that's routing or an agent — different patterns, covered in other docs in this series.

## Core concepts

- **Fixed decomposition** — the steps and their order are decided by the developer up front, not by the model at runtime.
- **Gate checks** — an optional but recommended programmatic check between steps (not another LLM call) that verifies the process is still on track before continuing — e.g. validating an outline's structure before drafting the full document from it.
- **Latency-for-accuracy tradeoff** — more steps means more round-trips and higher latency, but each individual call is simpler and less likely to fail at doing too much at once.

## Implementing the pattern

#### 1. Define the chain's steps
- **Action**: Write out an ordered list of step prompts — each one a focused instruction for what that stage of the chain should do.
  ```python
  steps = [
      "Write a structured outline for the following topic.",
      "Expand the outline below into a full first draft.",
      "Proofread and tighten the draft below for clarity and tone.",
  ]
  ```
- **Result**: `steps` defines the fixed sequence the chain will execute, in order, every time.

#### 2. Implement the chain loop
- **Action**: Feed each step's prompt, combined with the accumulated output from the previous step, to the LLM — the response becomes the new accumulated output.
  ```python
  def run_chain(steps: list[str], initial_input: str) -> str:
      response = initial_input
      for step_prompt in steps:
          prompt = f"{step_prompt}\n\n{response}"
          response = llm.invoke(prompt).content
      return response
  ```
- **Result**: `run_chain` walks through every step in order, each one building directly on the last step's output.

#### 3. Add a gate check
- **Action**: Insert a plain Python check — not another LLM call — that validates intermediate output before continuing, short-circuiting the chain if it fails.
  ```python
  def gate_check_outline(outline: str) -> None:
      sections = [line for line in outline.splitlines() if line.strip()]
      if len(sections) < 3:
          raise ValueError(
              f"Outline only has {len(sections)} section(s) — expected at least 3 before drafting."
          )
  ```
- **Result**: A malformed or too-thin outline stops the chain immediately, instead of silently propagating into a weak final draft.

#### 4. Run the chain end-to-end
- **Action**: Wire the pieces together — outline, gate check, then the rest of the chain.
  ```python
  outline = llm.invoke(
      f"Write a structured outline for the following topic.\n\n{topic}"
  ).content

  gate_check_outline(outline)

  remaining_steps = [
      "Expand the outline below into a full first draft.",
      "Proofread and tighten the draft below for clarity and tone.",
  ]
  final_draft = run_chain(remaining_steps, outline)
  ```
- **Result**: `final_draft` is the product of three focused LLM calls — outline, draft, polish — with a validation checkpoint after the first step, rather than one prompt asked to do all three at once.

> **Warning:**
> Common mistakes to check for:
> - **Forcing chaining onto a task that doesn't decompose cleanly** — if the steps aren't genuinely sequential and independent, chaining adds latency without an accuracy benefit.
> - **Skipping gate checks** — without them, a broken or malformed intermediate step propagates silently into the final output instead of failing fast.
> - **Too many chain steps** — every step is a full LLM round-trip; more steps than the task's actual complexity warrants just adds latency and cost.
> - **Reaching for chaining when the task needs dynamic branching** — a fixed chain can't adapt its own steps at runtime. That's routing or an agent, not this pattern.

## References

- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Spring AI — Effective Agents reference](https://docs.spring.io/spring-ai/reference/api/effective-agents.html)
- [Claude Code's five agentic patterns (real-world mapping, supporting context)](https://wmedia.es/en/tips/claude-code-agentic-ai-five-patterns)
