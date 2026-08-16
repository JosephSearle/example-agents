"""Unit tests for agents_common.prompts — pure logic, no live MLflow instance."""

from __future__ import annotations

from agents_common.prompts import link_prompts_to_trace, prompt_text
import pytest

pytestmark = pytest.mark.unit


class _FakePromptVersion:
    def __init__(self, template: object, name: str = "some-prompt") -> None:
        self.template = template
        self.name = name


def test_prompt_text_returns_plain_text_template() -> None:
    assert prompt_text(_FakePromptVersion("You are a helpful assistant.")) == (  # type: ignore[arg-type]
        "You are a helpful assistant."
    )


def test_prompt_text_raises_on_non_string_template() -> None:
    prompt_version = _FakePromptVersion([{"role": "system", "content": "hi"}], name="chat-prompt")

    with pytest.raises(TypeError, match="chat-prompt"):
        prompt_text(prompt_version)  # type: ignore[arg-type]


def test_link_prompts_to_trace_is_a_noop_without_a_trace_id() -> None:
    # No trace_id (e.g. autologging produced nothing to link against) shouldn't raise or attempt
    # any MLflow call — exercised without a live MLflow instance, unlike the "real" linking path.
    link_prompts_to_trace([object(), object()], trace_id=None)  # type: ignore[list-item]
