# Evaluator-Optimizer

One LLM generates a response while another evaluates it and provides feedback, in a loop — the same iterative process a human writer goes through drafting, getting feedback, and revising until the piece is polished.

![Evaluator-Optimizer workflow: a Task goes to a Generator producing a Draft, which an Evaluator judges; if it needs refinement, feedback loops back to the Generator for another iteration, and once it meets criteria it proceeds to Output](../../public/images/evaluator-optimizer.svg)

> **Note:**
> Per [Anthropic's framing](https://www.anthropic.com/engineering/building-effective-agents), this pattern is *"particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value."* Two signs it's a good fit: *"LLM responses can be demonstrably improved when a human articulates their feedback"*, and *"the LLM can provide such feedback"* itself. If neither holds — no clear way to judge "better," or the LLM can't meaningfully critique its own domain — this pattern just adds latency and cost for no measurable gain.

## Core concepts

- **Generate → evaluate → refine, looped** — the same three-part cycle as [Self-Refine](https://arxiv.org/abs/2303.17651), but split across separate calls rather than one model doing all three internally (see the distinction below).
- **Stopping condition** — the loop needs an explicit exit: the evaluator approves, or a maximum iteration count is hit. Never loop unboundedly.

> **Warning:**
> **Self-Refine vs. this implementation.** [Self-Refine (Madaan et al., NeurIPS 2023)](https://arxiv.org/abs/2303.17651) is genuinely **self**-feedback: one model plays generator, critic, and refiner on its own output, with no additional training required. Anthropic's evaluator-optimizer workflow — and the implementation below — typically splits generator and evaluator into **two separate LLM calls**, often even two different models (a stronger one evaluating a cheaper one's output). This is a deliberate production choice, not an inaccuracy: separate evaluation tends to be more reliable than a model critiquing itself, since the same blind spots that produced a flawed output can also affect that same model's self-assessment of it. What follows implements the two-call version, not Self-Refine's single-model mechanism.

## Implementing the pattern

#### 1. Define the evaluator's output shape
- **Action**: Structured output with a pass/fail verdict and specific feedback.
  ```python
  from pydantic import BaseModel, Field

  class Evaluation(BaseModel):
      approved: bool = Field(description="True if the response meets the criteria, false otherwise.")
      feedback: str = Field(description="Specific, actionable feedback for improving the response.")
  ```
- **Result**: `Evaluation` gives the loop both a stopping signal (`approved`) and something concrete to refine against (`feedback`).

#### 2. Generate an initial response
- **Action**: A first-pass generation, same as any other pattern in this series.
  ```python
  def generate(task: str, feedback: str | None = None) -> str:
      prompt = task if feedback is None else f"{task}\n\nPrevious feedback to address:\n{feedback}"
      return weak_model.invoke(prompt).content
  ```
- **Result**: `generate` produces a response — and, on later calls, incorporates the prior round's feedback.

#### 3. Evaluate the response
- **Action**: A separate LLM call — optionally a stronger model — scores the response against explicit criteria.
  ```python
  evaluator = strong_model.with_structured_output(Evaluation)

  def evaluate(task: str, response: str, criteria: str) -> Evaluation:
      return evaluator.invoke(
          f"Evaluate this response against the criteria below.\n\n"
          f"Task: {task}\n\nCriteria: {criteria}\n\nResponse: {response}"
      )
  ```
- **Result**: `evaluate` returns a verdict and specific feedback — a separate judgment, not the generator's own self-assessment.

#### 4. Refine using the feedback
- **Action**: Feed the evaluator's feedback back into another generation call.
  ```python
  def refine(task: str, evaluation: Evaluation) -> str:
      return generate(task, feedback=evaluation.feedback)
  ```
- **Result**: A revised response that directly addresses what the evaluator flagged, rather than a blind retry.

#### 5. Loop until approved or capped
- **Action**: Wrap generate/evaluate/refine in a loop with a hard iteration limit.
  ```python
  def evaluator_optimizer_loop(task: str, criteria: str, max_iterations: int = 3) -> str:
      response = generate(task)
      for _ in range(max_iterations):
          evaluation = evaluate(task, response, criteria)
          if evaluation.approved:
              return response
          response = refine(task, evaluation)
      return response
  ```
- **Result**: The loop exits as soon as the evaluator approves, or after `max_iterations` — never indefinitely.

#### 6. Worked example
- **Action**: Anthropic's own literary translation example — generate a translation, have the evaluator critique nuance and tone, refine, repeat.
  ```python
  task = "Translate the following passage into French, preserving tone and idiom."
  criteria = "Natural French phrasing; idioms adapted rather than translated literally; tone preserved."

  final_translation = evaluator_optimizer_loop(task, criteria)
  ```
- **Result**: A translation that's been through however many refinement rounds it needed — a first-pass translator often misses exactly the kind of nuance a dedicated evaluator pass is built to catch.

> **Warning:**
> Common mistakes to check for:
> - **No clear evaluation criteria** — if there's no concrete way to judge "better," the loop has nothing to converge toward and will just churn.
> - **Unbounded loops** — always cap iterations. An evaluator that's never satisfied will loop forever without a `max_iterations` guard.
> - **Cost multiplies with iteration count** — each round is at least two LLM calls (evaluate + refine), on top of the initial generation.
> - **Conflating this with genuine self-refinement** — if you actually want one model critiquing and refining its own output, that's [Self-Refine](https://arxiv.org/abs/2303.17651)'s exact mechanism, a different and cheaper single-model design with its own reliability tradeoffs — not what's implemented here.

## References

- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., NeurIPS 2023, arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)
- [Orchestrator-Workers](./orchestrator-workers)
- [Prompt Chaining](./prompt-chaining)
