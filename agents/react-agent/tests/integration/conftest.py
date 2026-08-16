"""Integration-test fixtures — require `docker compose up -d postgres` (see repo README)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents_common import get_settings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.postgres import PostgresSaver
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool


@pytest.fixture
def postgres_checkpointer() -> Iterator[PostgresSaver]:
    """A real PostgresSaver against the docker-compose Postgres instance.

    Skips (rather than fails) if `Settings.postgres_uri` isn't reachable, so `pytest -m
    integration` fails loudly in CI (where the service container is always up) but doesn't
    block a laptop run that hasn't started docker compose.
    """
    uri = get_settings().postgres_uri
    try:
        with PostgresSaver.from_conn_string(uri) as checkpointer:
            checkpointer.setup()
            yield checkpointer
    except Exception as exc:
        pytest.skip(f"Postgres not reachable at {uri}: {exc}")


class _FakeStructuredChatModel(BaseChatModel):
    """Always answers by calling the `AgentResponse` structured-output tool.

    Stands in for the real gateway model so checkpointing tests exercise real Postgres without
    depending on network access to a live model — this "integration" marker is about Postgres,
    the same philosophy `postgres_checkpointer` above already follows. Mirrors `ToolStrategy`'s
    synthetic tool name (the schema class's `__name__`, "AgentResponse" — see
    react_agent.graph.AgentResponse) so `create_agent`'s structured-output handling succeeds
    without ever calling a real model.
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "AgentResponse",
                    "args": {"answer": "Got it.", "used_tools": []},
                    "id": "fake-call-1",
                    "type": "tool_call",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-structured-chat-model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        """No-op: `_generate` ignores bound tools and always returns the same fake tool call."""
        return self


@pytest.fixture
def fake_chat_model(monkeypatch: pytest.MonkeyPatch) -> _FakeStructuredChatModel:
    """Patches `react_agent.graph.get_chat_model` so `build_agent()` never reaches the network."""
    model = _FakeStructuredChatModel()
    monkeypatch.setattr("react_agent.graph.get_chat_model", lambda _route: model)
    return model
