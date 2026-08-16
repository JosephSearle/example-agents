# RAG Adoption Strategy

This series covers six increasingly sophisticated RAG patterns — from [Basic RAG](./basic-rag) through [Adaptive RAG](./adaptive-rag). This doc isn't another pattern. It's how to decide which one(s) you actually need, and in what order, before writing any code.

> **Note:**
> There's no single paper behind this doc the way there is for [Corrective RAG](./corrective-rag-crag), [Self-RAG](./self-rag), or [Adaptive RAG](./adaptive-rag) — this is a synthesis of the patterns already documented in this series. The References section below is background reading on the broader RAG landscape, not the source this strategy was derived from.

## The three tiers

[LangChain's retrieval architecture guide](https://docs.langchain.com/oss/python/langchain/retrieval) frames the top-level architectural choice as three tiers. Every pattern in this series falls cleanly into one of them:

| Tier | What it means | Control / Flexibility / Latency | Example use case | Patterns in this series |
|---|---|---|---|---|
| **2-Step RAG** | Retrieval always runs before generation, in a fixed sequence | High control, low flexibility, fast & predictable latency | FAQs, documentation bots | [Basic RAG](./basic-rag), [Retrieve & Rerank](./retrieve-rerank) |
| **Hybrid RAG** | A fixed pipeline with added validation or query-enhancement steps | Moderate control/flexibility, variable but bounded latency | Ambiguous or underspecified queries needing refinement | [Corrective RAG](./corrective-rag-crag), [Self-RAG](./self-rag), [Query Decomposition](./query-decomposition) |
| **Agentic RAG** | An LLM agent decides when and how to retrieve, per query | Low control, high flexibility, variable latency | Research assistants with access to multiple tools | [Adaptive RAG](./adaptive-rag) |

[Retrieve & Rerank](./retrieve-rerank) stays in the 2-Step tier — reranking is a quality upgrade inside the same fixed retrieve-then-generate shape, not a new decision point. Everything in the Hybrid tier adds a validation or enhancement step ([Corrective RAG](./corrective-rag-crag) validates retrieval, [Self-RAG](./self-rag) validates retrieval and the generation itself, [Query Decomposition](./query-decomposition) enhances the query by breaking it apart) without handing control to an agent. Only [Adaptive RAG](./adaptive-rag) actually lets a model decide the retrieval strategy itself.

## Deciding what you need

#### 1. Start with Basic RAG
- Always the baseline. Don't reach for anything else in this series until you've actually run [Basic RAG](./basic-rag) and observed a specific shortfall — every pattern past it exists to fix a particular, diagnosable failure mode, not to be more sophisticated for its own sake.

#### 2. Is the problem precision within an already-decent retrieval set?
- If the right chunk is usually somewhere in your top-k, just ranked too low → [Retrieve & Rerank](./retrieve-rerank). Still 2-Step, the cheapest upgrade available, no architecture change.

#### 3. Is the problem retrieval quality itself?
- If irrelevant documents are regularly making it into context regardless of reranking → [Corrective RAG](./corrective-rag-crag), which grades retrieved documents and corrects course before generation.

#### 4. Is the problem ungrounded or hallucinated generations?
- If retrieval looks fine but the model's answers still aren't well-supported by what was retrieved → [Self-RAG](./self-rag), which checks the generation itself, not just the retrieved documents.

#### 5. Is the problem genuinely multi-part or multi-hop questions?
- If a single retrieval pass structurally can't cover what the question is actually asking → [Query Decomposition](./query-decomposition), which breaks the question into answerable sub-questions.

#### 6. Is the problem query mix heterogeneity?
- If some incoming queries need no retrieval at all and others need much more, and one fixed pipeline is either wasting cost on simple queries or under-serving complex ones → [Adaptive RAG](./adaptive-rag). Only take this step once you're willing to take on real routing complexity and cost — it's the one pattern in this series that hands a decision to an agent instead of following a fixed shape.

## Core guidance

Adopt in the order of the failure you've actually diagnosed, not in order of sophistication. [Gao et al.'s RAG survey](https://arxiv.org/abs/2312.10997) frames this as Naive → Advanced → Modular: start simple, add specific techniques only as needed, compose modularly rather than reaching for the most sophisticated option up front. Going straight to [Adaptive RAG](./adaptive-rag) or [Self-RAG](./self-rag) without first confirming what's actually wrong with [Basic RAG](./basic-rag) is premature complexity — it adds real latency and cost with nothing concrete to show for it.

> **Warning:**
> Common mistakes to check for:
> - **Guessing instead of diagnosing** — adopting a pattern without first measuring the specific failure mode it's meant to fix. Each step in the decision walkthrough above corresponds to a concrete, observable problem — confirm you actually have that problem first.
> - **Stacking hybrid patterns too early** — combining [Corrective RAG](./corrective-rag-crag), [Self-RAG](./self-rag), and [Query Decomposition](./query-decomposition) all at once before confirming each one individually pulls its weight, rather than layering them in one at a time.
> - **Treating Agentic RAG as a default** — [Adaptive RAG](./adaptive-rag) is a deliberate step up in operational complexity, not the "most advanced, so best" option. Reach for it only once query mix heterogeneity is a real, observed problem.
> - **Underestimating compounding cost** — every pattern past [Basic RAG](./basic-rag) adds at least one more LLM call per query. That cost compounds quickly if patterns are stacked without justification.

## References

Background reading on the broader RAG landscape — not the source this strategy is derived from; the adoption logic above synthesizes the patterns and papers already cited throughout this series.

- [Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey" (arXiv:2312.10997)](https://arxiv.org/abs/2312.10997)
- [LangChain — Retrieval architecture guide](https://docs.langchain.com/oss/python/langchain/retrieval)
- ["Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers" (arXiv:2506.00054, 2025)](https://arxiv.org/abs/2506.00054)
- [Basic RAG](./basic-rag)
- [Retrieve & Rerank](./retrieve-rerank)
- [Corrective RAG](./corrective-rag-crag)
- [Self-RAG](./self-rag)
- [Query Decomposition](./query-decomposition)
- [Adaptive RAG](./adaptive-rag)
