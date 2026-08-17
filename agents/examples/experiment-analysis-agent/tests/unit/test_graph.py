"""Unit tests for experiment_analysis_agent.graph — pure logic, no network, no MCP server, no LLM."""

from __future__ import annotations

from pathlib import Path

from experiment_analysis_agent.graph import (
    MLFLOW_MCP_TOOL_ALLOWLIST,
    REPORT_PATH,
    _filter_allowlisted_tools,
    _flatten_message_content,
    _flatten_text_only_content,
    render_system_prompt,
)
from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities.model_registry import PromptVersion
import pytest

pytestmark = pytest.mark.unit

# The git-tracked prompt template MLflow's registry is provisioned from — see
# packages/mlflow-server/scripts/provision_prompts.py. Loaded directly here (rather than via a
# live registry fetch) so this test is hermetic but still exercises the real template text.
_PROMPT_PATH = (
    Path(__file__).resolve().parents[5]
    / "packages"
    / "mlflow-server"
    / "prompts"
    / "experiment-analysis-agent.txt"
)


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


def test_filter_allowlisted_tools_keeps_only_read_only_mlflow_tools() -> None:
    tools = [
        _FakeTool(name)
        for name in (*MLFLOW_MCP_TOOL_ALLOWLIST, "delete_traces", "delete_experiment")
    ]

    filtered = _filter_allowlisted_tools(tools)  # type: ignore[arg-type]

    filtered_names = {tool.name for tool in filtered}
    assert filtered_names == set(MLFLOW_MCP_TOOL_ALLOWLIST)
    assert "delete_traces" not in filtered_names
    assert "delete_experiment" not in filtered_names


def test_filter_allowlisted_tools_handles_empty_input() -> None:
    assert _filter_allowlisted_tools([]) == []


def test_render_system_prompt_fills_in_target_experiment_and_report_path() -> None:
    prompt_version = PromptVersion(
        name="experiment-analysis-agent", version=1, template=_PROMPT_PATH.read_text()
    )

    rendered = render_system_prompt(prompt_version, target_experiment="routing-agent")

    assert "routing-agent" in rendered
    assert REPORT_PATH in rendered
    assert "{{" not in rendered, "expected every template variable to be filled in"


def test_flatten_text_only_content_collapses_single_text_block() -> None:
    assert _flatten_text_only_content([{"type": "text", "text": "hello"}]) == "hello"


def test_flatten_text_only_content_joins_multiple_text_blocks() -> None:
    blocks = [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]
    assert _flatten_text_only_content(blocks) == "hello world"


def test_flatten_text_only_content_leaves_plain_strings_alone() -> None:
    assert _flatten_text_only_content("already a string") == "already a string"


def test_flatten_text_only_content_leaves_non_text_blocks_alone() -> None:
    # e.g. an image block — this repo's gateway compatibility shim only handles the
    # text-blocks-list shape deepagents produces for system prompts, not multimodal content.
    blocks = [{"type": "image_url", "image_url": {"url": "data:..."}}]
    assert _flatten_text_only_content(blocks) == blocks


def test_flatten_message_content_flattens_a_system_message() -> None:
    message = SystemMessage(content=[{"type": "text", "text": "You are an assistant."}])

    flattened = _flatten_message_content(message)

    assert flattened.content == "You are an assistant."


def test_flatten_message_content_is_a_noop_for_plain_string_content() -> None:
    message = HumanMessage(content="hi")

    assert _flatten_message_content(message) is message
