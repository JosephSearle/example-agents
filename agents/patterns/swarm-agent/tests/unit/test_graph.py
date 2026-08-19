"""Unit tests for swarm_agent.graph — pure logic, no network, no LLM.

Mirrors react_agent's and supervisor_agent's own unit suites: building the actual swarm against
a stubbed model is exercised at the integration level (see
tests/integration/test_checkpointing.py, which uses a fake `BaseChatModel`), not here.
"""

from __future__ import annotations

import pytest
from swarm_agent.graph import DEFAULT_RECURSION_LIMIT, invoke_config, link_prompts_to_trace

pytestmark = pytest.mark.unit


def test_link_prompts_to_trace_is_a_noop_without_a_trace_id() -> None:
    # No trace_id (e.g. autologging produced nothing to link against) shouldn't raise or attempt
    # any MLflow call — exercised without a live MLflow instance, unlike the "real" linking path.
    link_prompts_to_trace({"triage": object()}, trace_id=None)  # type: ignore[dict-item]


def test_invoke_config_applies_default_recursion_limit() -> None:
    config = invoke_config("some-thread-id")

    assert config["configurable"]["thread_id"] == "some-thread-id"
    assert config["recursion_limit"] == DEFAULT_RECURSION_LIMIT


def test_invoke_config_accepts_recursion_limit_override() -> None:
    config = invoke_config("some-thread-id", recursion_limit=5)

    assert config["recursion_limit"] == 5
