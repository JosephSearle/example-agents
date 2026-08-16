"""Builds the ReAct agent graph.

`create_agent` (LangChain v1) compiles down to a LangGraph graph itself, so everything that
applies to "real" LangGraph graphs elsewhere in this repo — checkpointers, stores, streaming,
structured `interrupt()` — applies here too. This is tier 1 in the framework-tiering decision;
see docs/decisions/0001-tech-stack.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents_common import get_chat_model
from langchain.agents import create_agent
from pydantic import BaseModel, Field, ValidationError

from react_agent.prompts import SYSTEM_PROMPT
from react_agent.tools import TOOLS

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

# This agent's own MLflow experiment — see agents_common.observability.configure_mlflow for
# why this is a per-agent constant rather than a shared name pulled from .env.
EXPERIMENT_NAME = "react-agent"

# The MLflow AI Gateway route this agent calls — provisioned via
# packages/mlflow-server/scripts/provision_gateway_route.py against our self-hosted
# OpenAI-compatible model. See agents_common.models.get_chat_model.
GATEWAY_ROUTE = "gpt-oss-120b"


class AgentResponse(BaseModel):
    """Structured final response the agent must produce."""

    answer: str = Field(description="The direct answer to the user's question.")
    used_tools: list[str] = Field(
        default_factory=list,
        description="Names of the tools that were called while producing this answer.",
    )


def build_agent(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the ReAct agent.

    The caller owns the checkpointer's lifecycle (opening/closing the underlying connection),
    so this function takes no ownership of it — see `react_agent.__main__` for the
    `with get_checkpointer() as checkpointer:` usage against Postgres, and
    `tests/unit/conftest.py` for the `InMemorySaver()` used in fast tests.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration)
            or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route to call for this agent's model. Defaults to
            this package's own `GATEWAY_ROUTE`; overridable for tests against a different route.

    Returns:
        A compiled LangGraph graph, invoked with `{"messages": [...]}`.
    """
    model = get_chat_model(gateway_route)

    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=AgentResponse,
        checkpointer=checkpointer,
    )


def extract_response(result: dict[str, Any]) -> AgentResponse:
    """Recover the agent's structured answer, working around a self-hosted-model quirk.

    `create_agent`'s default `ToolStrategy` expects the model to emit a synthetic tool call for
    the final structured answer; `result["structured_response"]` is populated from that call.
    Our self-hosted OpenAI-compatible model calls *real* tools correctly but writes its final
    answer as plain-text JSON in the last message's content instead of via that synthetic tool
    call, leaving `structured_response` None. (`ProviderStrategy`, the natural fix for
    OpenAI-compatible native structured output, was tried and rejected — this backend can't
    combine a forced `response_format` with genuine tool-calling in the same turn, so it broke
    the tool-calling turns instead.) This parses that JSON content as a fallback. See
    docs/decisions/0001-tech-stack.md.

    Raises:
        ValueError: Neither a structured response nor parseable JSON content was found — a
            real failure, not this specific compatibility quirk.
    """
    structured_response = result.get("structured_response")
    if structured_response is not None:
        return structured_response  # type: ignore[no-any-return]

    last_message = result["messages"][-1]
    try:
        return AgentResponse.model_validate_json(last_message.content)
    except ValidationError as exc:
        msg = (
            "Agent produced neither a structured_response nor JSON-parseable final message "
            f"content: {last_message.content!r}"
        )
        raise ValueError(msg) from exc
