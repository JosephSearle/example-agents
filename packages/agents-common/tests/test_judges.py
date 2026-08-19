"""Unit tests for agents_common.judges — reads real, checked-in judge text files."""

from __future__ import annotations

from agents_common.judges import build_production_scorers, load_judge_guidelines
from mlflow.genai.scorers import Guidelines, Safety
import pytest

pytestmark = pytest.mark.unit


def test_load_judge_guidelines_reads_the_matching_file() -> None:
    text = load_judge_guidelines("react-agent-concise_answer")

    assert text == "The answer must be a direct response with no meta-commentary about tool usage."


def test_load_judge_guidelines_raises_on_a_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_judge_guidelines("no-such-agent-no-such-judge")


def test_build_production_scorers_appends_a_safety_scorer() -> None:
    scorers = build_production_scorers(
        "gpt-oss-120b", [("concise_answer", "react-agent-concise_answer")]
    )

    kinds = [type(scorer) for scorer, _sample_rate in scorers]
    assert kinds == [Guidelines, Safety]


def test_build_production_scorers_uses_the_gateway_model_form() -> None:
    (guidelines_scorer, guidelines_rate), (safety_scorer, safety_rate) = build_production_scorers(
        "gpt-oss-120b", [("concise_answer", "react-agent-concise_answer")]
    )

    assert guidelines_scorer.model == "gateway:/gpt-oss-120b"
    assert safety_scorer.model == "gateway:/gpt-oss-120b"
    assert guidelines_rate == safety_rate == 0.2


def test_build_production_scorers_loads_guideline_text_from_the_named_file() -> None:
    (guidelines_scorer, _rate), _safety = build_production_scorers(
        "gpt-oss-120b", [("concise_answer", "react-agent-concise_answer")]
    )

    assert (
        guidelines_scorer.guidelines
        == "The answer must be a direct response with no meta-commentary about tool usage."
    )


def test_build_production_scorers_supports_multiple_guidelines_entries() -> None:
    scorers = build_production_scorers(
        "gpt-oss-120b",
        [
            ("concise_answer", "react-agent-concise_answer"),
            ("relevant_response", "routing-agent-relevant_response"),
        ],
    )

    names = [scorer.name for scorer, _sample_rate in scorers[:-1]]
    assert names == ["concise_answer", "relevant_response"]
    assert len(scorers) == 3  # two Guidelines scorers + one trailing Safety scorer
