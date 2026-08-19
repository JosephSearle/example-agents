"""Integration-test fixtures — require `docker compose up -d postgres` (see repo README)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents_common import get_settings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import PrivateAttr
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


class _FakeChatModel(BaseChatModel):
    """Plays back a fixed sequence of `AIMessage`s, one per `.invoke()` call, in order.

    A single instance of this model is shared across the supervisor and both sub-agents (all
    three `create_agent()` calls in `build_supervisor` call `get_chat_model(gateway_route)`,
    which this fixture patches to always return this one instance) — since `create_agent`'s
    internal ReAct loop calls the model sequentially within one `.invoke()`, and sub-agent
    delegation is itself synchronous, the overall call order across supervisor and sub-agents is
    deterministic for a single top-level invocation, which is what this canned sequence relies
    on. Stands in for the real gateway model so checkpointing tests exercise real Postgres
    without depending on network access to a live model.
    """

    responses: list[AIMessage]
    _index: int = PrivateAttr(default=0)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self.responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-supervisor-chat-model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        """No-op: `_generate` ignores bound tools and plays back the canned sequence regardless."""
        return self


@pytest.fixture
def fake_chat_model(monkeypatch: pytest.MonkeyPatch) -> _FakeChatModel:
    """Patches `supervisor_agent.graph.get_chat_model` so `build_supervisor()` never reaches the
    network.

    Canned sequence: supervisor delegates to the math sub-agent, which calls `calculator` then
    answers "4"; the supervisor then gives its own final answer using that result. Four model
    calls total, in that exact order.
    """
    model = _FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delegate_to_math",
                        "args": {"request": "What is 2+2?"},
                        "id": "fake-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "2+2"},
                        "id": "fake-call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="4"),
            AIMessage(content="The answer is 4."),
        ]
    )
    monkeypatch.setattr("supervisor_agent.graph.get_chat_model", lambda _route: model)
    return model


@pytest.fixture
def agent_prompts() -> dict[str, str]:
    """Hermetic agent prompts, so tests don't depend on a live MLflow prompt registry."""
    return {
        "supervisor": "You delegate math requests to delegate_to_math.",
        "math": "You use the calculator tool to answer arithmetic requests.",
        "text": "You use count_words/reverse_text to answer text requests.",
    }
