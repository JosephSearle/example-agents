# ADR 0002: Eval taxonomy — grader types, capability vs regression evals, and CI gating

- **Status:** Accepted
- **Date:** 2026-08-20
- **Owner:** Joseph Searle

## Context

Every pattern's `tests/evals/test_quality.py` tested one facet only: answer quality, via an
LLM-judge `Guidelines`/`Correctness` scorer, occasionally paired with one hand-rolled
deterministic scorer (`correct_route`, `delegated_to_the_required_sub_agents`,
`correct_active_agent`). Nothing covered tool-call/trajectory correctness as a first-class
concept, safety/adversarial inputs, non-determinism across repeated runs, or a fast,
CI-required regression gate — `.github/workflows/eval.yml` runs nightly or on-demand only, by
design (see ADR 0001's stack reasoning: don't spend tokens on every PR), which left no
CI-required signal at all for "did this change break something that used to work."

Anthropic's [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
gives this repo a taxonomy worth standardizing on, since this is a reference repo and the
*shape* of the eval suite is as much the deliverable as the agents themselves.

## Decision

### Grader types

Three grader types, each already represented somewhere in this repo:

| Type | Examples in this repo | Where |
|---|---|---|
| Code-based | `correct_route`, `delegated_to_the_required_sub_agents`, `correct_active_agent`, the new `tool_calls_include` factory | `agents_common.judges`, per-pattern `test_quality.py` |
| Model-based | `Guidelines`, `Correctness`, `Safety` | `mlflow.genai.scorers`, wired through `agents_common.judges` |
| Human | MLflow's evaluation-runs UI, spot-checked manually | Not automated — intentionally: this is a reference repo, not a product with a labeling pipeline |

Code-based graders are preferred wherever a deterministic check is possible (trajectory,
routing, delegation) — they're fast, free, and don't need calibration. Model-based graders
cover what only a judge can assess (groundedness, tone, safety). This is why every pattern's
regression suite (below) is code-based-grader-first.

### Capability evals vs regression evals

Two distinct suites per pattern now, not one:

- **Capability evals** (`-m eval`, existing) — the full dataset, LLM-judge-heavy, threshold
  `>= 0.7`–`0.8`. Answers "how good is this agent," which is inherently noisy and improves
  incrementally — not something to gate a PR on. Stays nightly/label-gated
  (`.github/workflows/eval.yml`), unchanged.
- **Regression evals** (`-m regression`, new) — a small, curated subset of each dataset, tagged
  `"tags": ["regression"]` in `packages/mlflow-server/datasets/<pattern>.jsonl`: previously
  verified-good cases, code-based-grader-first, threshold `>= 0.95`. Answers "did this change
  break something that used to work" — small and stable enough to require on every PR
  (`.github/workflows/regression.yml`). This is the gap ADR 0001 left open, closed without
  reintroducing "every PR calls a real model expensively."

Regression suites reuse each pattern's existing `_predict_fn` and dataset rather than
maintaining a parallel dataset or MLflow experiment — new rows are added to the same JSONL,
just flagged with the `regression` tag.

### Trajectory scoring

`agents_common.judges.tool_calls_include(name, trajectory_key, expected_key)` generalizes the
subset-match pattern `supervisor-agent` and `swarm-agent` already used ad hoc: does the observed
trajectory (tool calls, delegate calls, active agent) include everything `expectations` requires,
allowing extra/retry calls a rigid exact-match would wrongly fail. Every pattern's eval suite
now has one such scorer, even the previously-untested ones (react-agent's calculator call,
corrective-rag-agent's retry decision, etc.) — a graph's own routing/trajectory field is enough
to check against, no new instrumentation required.

### Safety and adversarial coverage

`Safety()` (already used in `agents_common.judges.build_production_scorers` for sampled
production traces) is now also included in every `test_quality.py`'s capability-eval scorer
list. Each dataset gained 1–2 adversarial/edge-case rows (prompt injection, off-topic ask,
empty/ambiguous input) so that scorer has something to actually catch, instead of running
against only well-formed questions.

### Non-determinism (pass@k / pass^k)

`agents_common.judges.pass_at_k` / `pass_hat_k` run `predict_fn` `k` times and report the
fraction that succeeded (`pass@k`) or whether *all* of them did (`pass^k`). Demonstrated in
exactly one place — `agents/patterns/react-agent/tests/evals/test_quality.py`, which already had
a documented (but previously unused) flakiness note about gpt-oss-120b intermittently skipping
the calculator tool call. Not applied blanket to every pattern: copy it only where a pattern has
*observed* non-determinism, not preemptively.

## Consequences

- `pytest -m regression` is a required GitHub Actions check (`regression.yml`); `pytest -m eval`
  stays nightly/label-gated (`eval.yml`) — same live-model dependency, different cost/frequency
  trade-off.
- Every pattern's dataset JSONL and `test_quality.py` follow one consistent shape now: capability
  test (existing), regression test (new, tag-filtered), trajectory scorer, `Safety` scorer.
- Maintaining this means every future pattern needs a `regression`-tagged subset from day one,
  not bolted on later — see the `new-agent-pattern` skill / react-agent reference layout.
