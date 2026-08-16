# Routing

Routing classifies an input first, then sends it down whichever specialized path actually fits — rather than handling every input the same way. There are two distinct reasons to reach for this: routing to a *specialized* prompt/tool for a given category of task, or routing to a *cheaper or stronger model* purely for cost and performance reasons.

![Routing workflow: an Input Query is classified, then sent down exactly one of three specialist paths (General Questions, Refund Requests, Technical Support), each leading to its own Response; only one path is active per query](../../../public/images/agent/routing.svg)

> **Note:**
> Like [Prompt Chaining](./prompt-chaining), this is a **workflow**, not an agent: the set of possible routes is fixed in code ahead of time. The classifier picks among predefined routes — it doesn't invent new ones at runtime.

## Core concepts

- **Classification first, action second** — every routing implementation has the same two-part shape: classify the input, then dispatch to whatever that classification implies.
- **Two distinct flavors** — **task routing**, where different categories genuinely need different handling (different prompts, tools, or context), and **model-tier routing**, where the handling is the same but the model doing it should vary by difficulty, purely for cost/latency reasons.
- **Classifier choice** — an LLM call with structured output, a traditional ML classifier, or a purpose-trained/signal-based router (per RouteLLM and semantic-router below) — pick based on how much the classification itself costs relative to what it's protecting against.

## Implementing the pattern

### Task Routing

Route different categories of task to genuinely different handling — separate prompts, tools, or context per category.

#### 1. Define the categories
- **Action**: Enumerate the distinct task types your router needs to distinguish between.
  ```python
  from pydantic import BaseModel, Field
  from typing import Literal

  class TicketCategory(BaseModel):
      category: Literal["general", "refund", "technical"] = Field(
          description="The category that best matches this support ticket."
      )
  ```
- **Result**: `TicketCategory` defines the fixed set of routes the classifier can choose between.

#### 2. Classify the input
- **Action**: Use an LLM with structured output to pick one category.
  ```python
  classifier = llm.with_structured_output(TicketCategory)

  def classify_ticket(message: str) -> str:
      result = classifier.invoke(message)
      return result.category
  ```
- **Result**: `classify_ticket` returns exactly one of `"general"`, `"refund"`, or `"technical"` for any incoming message.

#### 3. Dispatch to the specialized handler
- **Action**: Map each category to its own handler — a different prompt, and potentially different tools, per category.
  ```python
  def handle_general(message: str) -> str:
      return llm.invoke(f"Answer this general support question:\n\n{message}").content

  def handle_refund(message: str) -> str:
      return llm.invoke(
          f"Follow the refund policy strictly. Ticket:\n\n{message}"
      ).content

  def handle_technical(message: str) -> str:
      return llm.invoke(
          f"Provide a technical troubleshooting response. Ticket:\n\n{message}"
      ).content

  handlers = {
      "general": handle_general,
      "refund": handle_refund,
      "technical": handle_technical,
  }
  ```
- **Result**: `handlers` maps each category to the specialized logic that category actually needs.

#### 4. Run it end-to-end
- **Action**: Classify, then dispatch.
  ```python
  def route_ticket(message: str) -> str:
      category = classify_ticket(message)
      return handlers[category](message)
  ```
- **Result**: `route_ticket` sends every incoming ticket through classification once, then straight to the handler built for that category — no handler needs to account for the other categories' concerns.

### Model-Tier Routing

Route to a cheaper or stronger model based on how difficult the query actually is — the handling stays the same, only the model changes.

> **Warning:**
> [RouteLLM (Ong et al., 2024)](https://arxiv.org/abs/2406.18665) is the primary academic paper for this flavor — it trains a router on human preference data to decide, per query, whether a weak/cheap model suffices or a strong/expensive model is needed, reporting over 2x cost reduction with minimal quality loss. Its own reference repo hasn't been meaningfully updated since August 2024, though — the paper's ideas are still foundational and widely cited, but production teams are largely reimplementing the idea themselves or reaching for actively-maintained tooling rather than depending on that repo directly. The example below is an original, from-scratch illustration of the same idea — not a port of RouteLLM's or [semantic-router](https://github.com/vllm-project/semantic-router)'s actual code.

#### 1. Define your model tiers
- **Action**: Set up a cheap/fast model and a strong/expensive model.
  ```python
  from langchain_anthropic import ChatAnthropic

  weak_model = ChatAnthropic(model="claude-haiku-4-5")
  strong_model = ChatAnthropic(model="claude-sonnet-4-6")
  ```
- **Result**: Two models with a real cost/capability gap between them, ready to be chosen between per query.

#### 2. Classify query difficulty
- **Action**: A lightweight step estimating whether the strong model is actually needed — cheap enough that it doesn't erode the savings routing is meant to produce.
  ```python
  class DifficultyRoute(BaseModel):
      needs_strong_model: bool = Field(
          description="True if this query requires complex reasoning, ambiguity resolution, or high stakes; false if it's simple/common."
      )

  difficulty_classifier = weak_model.with_structured_output(DifficultyRoute)

  def classify_difficulty(query: str) -> bool:
      return difficulty_classifier.invoke(query).needs_strong_model
  ```
- **Result**: The classification itself runs on the cheap model — it only needs to be cheap and directionally right, not perfect.

#### 3. Route to the appropriate tier
- **Action**: Dispatch to the weak or strong model based on that classification.
  ```python
  def route_by_difficulty(query: str) -> str:
      model = strong_model if classify_difficulty(query) else weak_model
      return model.invoke(query).content
  ```
- **Result**: Easy, common queries are answered by the cheap model; only queries judged to need it pay the cost of the strong model.

#### 4. Where to go for a production-grade version
- This simple classifier is a starting point, not a finished system. [RouteLLM](https://arxiv.org/abs/2406.18665) trains a dedicated preference-based router specifically for this decision (though, per the caveat above, its own repo isn't actively maintained). [semantic-router](https://github.com/vllm-project/semantic-router) implements this at the infrastructure layer instead — a signal-driven router that combines domain/embedding-based classification with safety plugins (jailbreak and PII detection, semantic caching) rather than a single trained preference classifier. Both are worth evaluating before building this out further in-house.

> **Warning:**
> Common mistakes to check for:
> - **Misclassification cost differs by flavor** — a wrong task-route sends the user down an entirely wrong specialized path; a wrong model-tier route just costs more or produces a slightly weaker answer. Calibrate how much you trust the classifier to the actual cost of it being wrong.
> - **The classifier itself has a cost** — for model-tier routing specifically, if the classification step costs nearly as much as just calling the strong model every time, the pattern isn't paying for itself.
> - **Routing isn't a substitute for chaining or agents** — if the real need is sequential steps or dynamic branching mid-task, that's [Prompt Chaining](./prompt-chaining) or an agent, not a one-shot category decision.
> - **Depending on an unmaintained reference implementation** — don't wire RouteLLM's original repo directly into production; reimplement the idea yourself or use an actively maintained tool.

## References

- [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [RouteLLM (Ong et al., 2024, arXiv:2406.18665)](https://arxiv.org/abs/2406.18665) — reference implementation last meaningfully updated August 2024
- [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router) — current, actively-maintained implementation reference
- [Prompt Chaining](./prompt-chaining)
