# Parallelization

Parallelization runs multiple LLM calls at the same time instead of one after another, then combines their outputs programmatically — either for speed (independent pieces of work run concurrently) or for confidence (multiple attempts at the same work, aggregated into one answer).

![Parallelization workflow with two variations: Sectioning splits a task into three independent subtasks that run simultaneously and are aggregated; Voting runs three identical attempts of the same task simultaneously and keeps the consensus result](../../../public/images/agent/parallelization.svg)

> **Note:**
> Like [Prompt Chaining](./prompt-chaining) and [Routing](./routing), this is a **workflow**, not an agent: which calls run and how their outputs are combined is fixed in code ahead of time — nothing here is dynamically decided by an LLM mid-flight.

## Core concepts

- **Sectioning vs. voting** — **sectioning** splits one task into independent pieces run concurrently (each piece is *different* work); **voting** runs the *same* task multiple times to get diverse attempts at *the same* work.
- **Aggregation is programmatic** — a fixed rule (majority vote, all-must-pass, an LLM-as-judge selection step) combines the parallel outputs. The combination step itself isn't dynamic branching.
- **Diversity is the point, in voting specifically** — running the same prompt N times only helps if the attempts are meaningfully diverse (temperature, phrasing variation). N near-identical, near-deterministic calls buy nothing.

## Implementing the pattern

### Sectioning

Split one task into genuinely independent pieces and run them concurrently — for speed, or to separate concerns (e.g. a safety check running alongside the actual response).

#### 1. Define independent subtasks
- **Action**: Write each subtask so it has no shared state or dependency on the others — each can run in complete isolation.
  ```python
  async def check_guardrail(user_input: str) -> bool:
      result = await llm.ainvoke(
          f"Does this input violate content policy? Answer yes or no.\n\n{user_input}"
      )
      return "yes" in result.content.lower()

  async def generate_response(user_input: str) -> str:
      result = await llm.ainvoke(f"Respond to the user:\n\n{user_input}")
      return result.content
  ```
- **Result**: Two calls that can run in either order, or at the same time, with no risk of one needing the other's output.

#### 2. Run them concurrently
- **Action**: Use `asyncio.gather` to fire both calls at once instead of waiting for one to finish before starting the next.
  ```python
  import asyncio

  async def run_sectioned(user_input: str):
      is_blocked, response = await asyncio.gather(
          check_guardrail(user_input),
          generate_response(user_input),
      )
      return is_blocked, response
  ```
- **Result**: Both calls run in parallel — total latency is roughly the slower of the two, not the sum of both.

#### 3. Aggregate results programmatically
- **Action**: Combine the outputs with a fixed rule, decided in code, not by another LLM call.
  ```python
  def aggregate_sectioned(is_blocked: bool, response: str) -> str:
      if is_blocked:
          return "I can't help with that request."
      return response
  ```
- **Result**: The guardrail's verdict deterministically overrides the generated response when it fires, regardless of what the response call produced.

#### 4. Worked example
- **Action**: Put it together — the guardrail check and the actual response run concurrently, and the guardrail wins if it flags anything.
  ```python
  async def handle_request(user_input: str) -> str:
      is_blocked, response = await run_sectioned(user_input)
      return aggregate_sectioned(is_blocked, response)
  ```
- **Result**: A request that would otherwise pay for the guardrail check and the response sequentially now pays for only the slower of the two.

### Voting

Run the *same* task multiple times to get diverse independent attempts, then aggregate into one higher-confidence answer.

> **Warning:**
> [Self-Consistency (Wang et al., 2022)](https://arxiv.org/abs/2203.11171) and its follow-on [Universal Self-Consistency](https://arxiv.org/abs/2311.17311) are about **a single model sampling multiple reasoning paths at the decoding level**, then majority-voting (or, for free-form answers, using the model itself to pick the most consistent sample) — not multiple agents or multiple independent orchestration-level calls. What follows **generalizes the same statistical idea** — diverse independent attempts plus consensus aggregation — up to a multi-call orchestration layer. The papers themselves describe single-model, multi-sample decoding, not the multi-call pattern below.

#### 1. Sample multiple diverse attempts
- **Action**: Run the same prompt N times concurrently, with enough temperature to get genuinely different attempts rather than N near-identical outputs.
  ```python
  import asyncio

  async def sample_attempts(prompt: str, n: int = 5) -> list[str]:
      results = await asyncio.gather(
          *[llm.ainvoke(prompt, temperature=0.8) for _ in range(n)]
      )
      return [r.content for r in results]
  ```
- **Result**: `attempts` holds N diverse independent responses to the same prompt.

#### 2. Aggregate via majority vote
- **Action**: For answers that extract cleanly to a discrete value, count occurrences and take the most common one — the self-consistency approach.
  ```python
  from collections import Counter

  def majority_vote(attempts: list[str]) -> str:
      counts = Counter(attempts)
      return counts.most_common(1)[0][0]
  ```
- **Result**: The answer most attempts converged on independently, rather than whatever a single sample happened to produce.

#### 3. Aggregate via LLM-as-judge consistency selection
- **Action**: For free-form answers where extraction doesn't work, use the model itself to identify the most representative response among the samples — the Universal Self-Consistency approach.
  ```python
  def usc_select(attempts: list[str]) -> str:
      numbered = "\n\n".join(f"[{i}] {a}" for i, a in enumerate(attempts))
      judge_prompt = f"""Below are {len(attempts)} candidate responses to the same
  question. Pick the one that best represents the consensus/most common answer
  among them, and return only its number.

  {numbered}"""
      choice = int(llm.invoke(judge_prompt).content.strip())
      return attempts[choice]
  ```
- **Result**: A consistency-based selection that works even when the answers are prose, not a clean extractable value.

#### 4. Worked example
- **Action**: Anthropic's own example — N independent code-vulnerability reviews, aggregated by an any-flag vote rather than majority (a single reviewer catching a real issue should still block).
  ```python
  async def review_for_vulnerabilities(code: str, n: int = 5) -> bool:
      reviews = await sample_attempts(
          f"Does this code contain a security vulnerability? Answer yes or no.\n\n{code}",
          n=n,
      )
      return any("yes" in review.lower() for review in reviews)
  ```
- **Result**: The code is flagged if *any* of the N independent reviews raises a concern — appropriate here because a missed vulnerability is more costly than a false positive.

> **Warning:**
> Common mistakes to check for:
> - **Cost multiplies directly with parallel calls** — N sectioning subtasks or N voting samples means N× the LLM cost of a single call. Running them concurrently buys latency and/or confidence, not cost savings.
> - **Sectioning requires genuine independence** — if subtasks secretly share state or depend on each other's output, running them in parallel produces wrong or inconsistent results.
> - **Voting only helps when errors are uncorrelated** — if the model is systematically wrong about something, running it N times just produces N confidently wrong answers, and majority vote won't fix that.
> - **LLM-as-judge aggregation is itself another LLM call** — the USC-style selection step stacks its own cost on top of the N samples it's evaluating.

## References

- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022, arXiv:2203.11171)](https://arxiv.org/abs/2203.11171)
- [Universal Self-Consistency (arXiv:2311.17311)](https://arxiv.org/abs/2311.17311)
- [Prompt Chaining](./prompt-chaining)
- [Routing](./routing)
