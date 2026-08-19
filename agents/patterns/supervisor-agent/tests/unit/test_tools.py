"""Unit tests for supervisor_agent.tools — pure functions, no network, no LLM."""

from __future__ import annotations

from hypothesis import given, strategies as st
import pytest
from supervisor_agent.tools import calculator, count_words, reverse_text

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        pytest.param("47 * 12", "564", id="multiplication"),
        pytest.param("(3 + 4) / 2", "3.5", id="mixed-precedence"),
        pytest.param("2 ** 10", "1024", id="exponent"),
        pytest.param("-5 + 2", "-3", id="unary-minus"),
    ],
)
def test_calculator_evaluates_arithmetic(expression: str, expected: str) -> None:
    assert calculator.invoke({"expression": expression}) == expected


def test_calculator_rejects_non_arithmetic_input() -> None:
    result = calculator.invoke({"expression": "__import__('os').system('echo pwned')"})
    assert "Could not evaluate" in result


def test_calculator_reports_division_by_zero() -> None:
    result = calculator.invoke({"expression": "1 / 0"})
    assert "Could not evaluate" in result


@given(
    a=st.integers(min_value=-1_000, max_value=1_000),
    b=st.integers(min_value=-1_000, max_value=1_000),
)
def test_calculator_addition_matches_python(a: int, b: int) -> None:
    result = calculator.invoke({"expression": f"{a} + {b}"})
    assert int(result) == a + b


def test_count_words_counts_whitespace_separated_words() -> None:
    assert count_words.invoke({"text": "the quick brown fox"}) == "4"


def test_count_words_empty_string_is_zero() -> None:
    assert count_words.invoke({"text": ""}) == "0"


def test_reverse_text_reverses_characters() -> None:
    assert reverse_text.invoke({"text": "hello"}) == "olleh"


def test_reverse_text_is_its_own_inverse() -> None:
    text = "a palindrome test"
    assert reverse_text.invoke({"text": reverse_text.invoke({"text": text})}) == text
