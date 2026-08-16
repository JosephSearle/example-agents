"""Unit tests for react_agent.graph.extract_response — pure logic, no network, no LLM."""

from __future__ import annotations

from langchain_core.messages import AIMessage
import pytest
from react_agent.graph import AgentResponse, extract_response

pytestmark = pytest.mark.unit


def test_extract_response_prefers_structured_response() -> None:
    result = {
        "messages": [AIMessage(content="ignored")],
        "structured_response": AgentResponse(answer="564", used_tools=["calculator"]),
    }

    response = extract_response(result)

    assert response.answer == "564"
    assert response.used_tools == ["calculator"]


def test_extract_response_falls_back_to_json_content() -> None:
    result = {
        "messages": [AIMessage(content='{"answer": "564", "used_tools": ["calculator"]}')],
        "structured_response": None,
    }

    response = extract_response(result)

    assert response.answer == "564"
    assert response.used_tools == ["calculator"]


def test_extract_response_json_content_without_used_tools_defaults_to_empty_list() -> None:
    result = {
        "messages": [AIMessage(content='{"answer": "564"}')],
        "structured_response": None,
    }

    response = extract_response(result)

    assert response.answer == "564"
    assert response.used_tools == []


def test_extract_response_raises_on_unparseable_content() -> None:
    result = {
        "messages": [AIMessage(content="not json at all")],
        "structured_response": None,
    }

    with pytest.raises(ValueError, match="neither a structured_response"):
        extract_response(result)


def test_extract_response_raises_on_json_missing_required_field() -> None:
    result = {
        "messages": [AIMessage(content="{}")],
        "structured_response": None,
    }

    with pytest.raises(ValueError, match="neither a structured_response"):
        extract_response(result)
