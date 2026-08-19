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

from mlflow.genai.scorers import Guidelines, Safety

if TYPE_CHECKING:
    from collections.abc import Sequence

_JUDGES_DIR = Path(__file__).resolve().parents[3] / "mlflow-server" / "judges"

_DEFAULT_SAMPLE_RATE = 0.2


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
