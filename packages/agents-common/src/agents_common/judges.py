"""Shared loader and builder for judge (`Guidelines`/`Safety` scorer) config.

Every agent's `PRODUCTION_SCORERS` and `tests/evals/test_quality.py` previously inlined its
`Guidelines` scorer's guideline text as a Python string literal, duplicated between the two call
sites (see e.g. routing_agent.graph's old `relevant_response` text, copy-pasted into its eval
suite). `packages/mlflow-server/judges/<name>.txt` is now the single source of truth, read via
`load_judge_guidelines` from both places — mirrors `agents_common.prompts`' role for prompt text,
but reads a plain file rather than fetching from MLflow's prompt registry: guideline text isn't
versioned/aliased/traced the way a system or step prompt is, so there's no registry round-trip to
make here.

Resolved relative to this file's own location (`packages/agents-common/src/agents_common/`, three
parents up to `packages/`) rather than any `PROJECT_ROOT`-style env var, so it works identically
in a local dev checkout and inside a built agent's Docker image — each agent's Dockerfile copies
its own judge text file(s) to the matching `packages/mlflow-server/judges/` path (see
`agents/patterns/*/Dockerfile`), so this resolves without needing packages/mlflow-server's other
contents (datasets, prompts, scripts) shipped into the image.

`build_production_scorers` goes one step further: every agent's `PRODUCTION_SCORERS` was also
duplicating the *boilerplate* around that text — deriving `gateway:/<route>` from `GATEWAY_ROUTE`
(see `agents/patterns/react-agent/src/react_agent/graph.py`'s `_MONITOR_JUDGE_MODEL_URI` docstring
for why `Scorer.start()` requires this form specifically), wrapping each guideline in a
`Guidelines(...)`, and appending a same-model `Safety(...)` scorer, all at a flat 0.2 sample rate.
None of that varies per agent today, so it's centralized here too — each agent's `graph.py` now
only declares which named guideline judges apply.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlflow.genai.scorers import Guidelines, Safety, scorer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_JUDGES_DIR = Path(__file__).resolve().parents[3] / "mlflow-server" / "judges"

_DEFAULT_SAMPLE_RATE = 0.2

# Re-exported so eval suites pull the safety scorer from the same module as everything else
# eval-related, instead of reaching into `mlflow.genai.scorers` directly.
__all__ = [
    "Safety",
    "build_production_scorers",
    "load_judge_guidelines",
    "pass_at_k",
    "pass_hat_k",
    "regression_subset",
    "tool_calls_include",
]


def load_judge_guidelines(name: str) -> str:
    """Read one judge's guideline text from `packages/mlflow-server/judges/<name>.txt`.

    Args:
        name: The judge's file stem, e.g. `"react-agent-concise_answer"`.
    """
    return (_JUDGES_DIR / f"{name}.txt").read_text().strip()


def build_production_scorers(
    gateway_route: str,
    guidelines: Sequence[tuple[str, str]],
    *,
    sample_rate: float = _DEFAULT_SAMPLE_RATE,
) -> list[tuple[Any, float]]:
    """Build a `PRODUCTION_SCORERS` list: one `Guidelines` scorer per entry, plus `Safety`.

    Every scorer shares the same judge model (`gateway:/<gateway_route>`, the form
    `Scorer.register().start()` requires — see this module's docstring) and the same
    `sample_rate`, since no agent in this repo varies either per scorer today.

    Args:
        gateway_route: The agent's own `GATEWAY_ROUTE`, e.g. `"gpt-oss-120b"`.
        guidelines: `(scorer_name, judge_file_stem)` pairs, one per `Guidelines` scorer to build.
            `judge_file_stem` is passed straight to `load_judge_guidelines`, e.g.
            `("coherent_report", "parallelization-agent-coherent_report")`.
        sample_rate: Fraction of production traces (0.0-1.0) each scorer runs against.

    Returns:
        A `list[tuple[Scorer, float]]`, ready to assign directly to an agent's
        `PRODUCTION_SCORERS`.
    """
    model_uri = f"gateway:/{gateway_route}"
    scorers: list[tuple[Any, float]] = [
        (
            Guidelines(
                name=scorer_name,
                guidelines=load_judge_guidelines(judge_name),
                model=model_uri,
            ),
            sample_rate,
        )
        for scorer_name, judge_name in guidelines
    ]
    scorers.append((Safety(model=model_uri), sample_rate))  # type: ignore[no-untyped-call]
    return scorers


def regression_subset(dataset: Any, *, tag: str = "regression") -> list[dict[str, Any]]:
    """Filter an `EvaluationDataset` down to rows tagged `tag` in their source JSONL.

    Regression suites reuse the same dataset/experiment as the capability eval suite rather than
    provisioning a parallel one — see `docs/decisions/0002-eval-taxonomy.md`. Rows opt in via a
    `"tags": ["regression"]` field on the JSONL record (`packages/mlflow-server/datasets/*.jsonl`),
    merged in as-is by `provision_datasets.py`.
    """
    records: list[dict[str, Any]] = dataset.to_df().to_dict("records")
    return [r for r in records if tag in (r.get("tags") or [])]


def tool_calls_include(name: str, *, trajectory_key: str, expected_key: str) -> Any:
    """Build a deterministic `@scorer` checking observed tool/agent calls against expectations.

    Subset match, not exact match: an agent re-delegating or retrying is still correct as long
    as every *required* name shows up somewhere in the trajectory. Generalizes the ad hoc
    `delegated_to_the_required_sub_agents` (supervisor-agent) and `correct_active_agent`
    (swarm-agent) scorers that previously reimplemented this per pattern.

    Args:
        name: The scorer's display name in MLflow eval results.
        trajectory_key: Key in `predict_fn`'s output dict holding the observed names, e.g.
            `"tool_calls"` or `"delegates"` — must be an iterable of strings.
        expected_key: Key in the dataset row's `expectations` dict holding the required names,
            e.g. `"expected_tool_calls"` — must be an iterable of strings.

    Returns:
        A `@scorer`-decorated function ready to pass into `mlflow.genai.evaluate(scorers=[...])`.
    """

    @scorer(name=name)
    def _scorer(outputs: dict[str, object], expectations: dict[str, object]) -> bool:
        observed = set(outputs[trajectory_key])  # type: ignore[call-overload]
        required = set(expectations[expected_key])  # type: ignore[call-overload]
        return bool(required.issubset(observed))

    return _scorer


def pass_at_k(
    predict_fn: Callable[[str], dict[str, object]],
    question: str,
    *,
    k: int,
    is_success: Callable[[dict[str, object]], bool],
) -> float:
    """Run `predict_fn(question)` `k` times, return the fraction that succeeded (pass@k).

    For flagging non-determinism in a single agent behavior (e.g. an LLM intermittently skipping
    a tool call) — see `agents/patterns/react-agent/tests/evals/test_quality.py` for the one
    worked example in this repo. Copy this only where a pattern has *observed* flakiness; it
    isn't meant to run on every pattern by default.
    """
    trials = [predict_fn(question) for _ in range(k)]
    return sum(1 for t in trials if is_success(t)) / k


def pass_hat_k(
    predict_fn: Callable[[str], dict[str, object]],
    question: str,
    *,
    k: int,
    is_success: Callable[[dict[str, object]], bool],
) -> float:
    """Run `predict_fn(question)` `k` times, return 1.0 only if *all* `k` trials succeeded (pass^k).

    The stricter counterpart to `pass_at_k` — use when consistency across repeated runs matters
    more than "got it right at least once."
    """
    trials = [predict_fn(question) for _ in range(k)]
    return 1.0 if all(is_success(t) for t in trials) else 0.0
