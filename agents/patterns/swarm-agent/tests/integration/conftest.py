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

    A single instance of this model is shared across both peer agents (both `create_agent()`
    calls in `build_swarm` call `get_chat_model(gateway_route)`, which this fixture patches to
    always return this one instance). Stands in for the real gateway model so checkpointing
    tests exercise real Postgres — including `create_swarm`'s own persistent-active-agent state
    across two turns in one thread — without depending on network access to a live model.
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
        return "fake-swarm-chat-model"

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
    """Patches `swarm_agent.graph.get_chat_model` so `build_swarm()` never reaches the network.

    Canned sequence, across two turns in the same thread:
    - Turn 1 ("refund INV-1002"): triage transfers to billing, billing calls `issue_refund`,
      billing gives its final answer.
    - Turn 2 ("what about INV-1001?"), same thread: the swarm resumes directly with billing (no
      transfer call needed — it's still the active agent), billing calls `lookup_invoice`,
      billing gives its final answer.

    Five model calls total, in that exact order.
    """
    model = _FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "transfer_to_billing",
                        "args": {},
                        "id": "fake-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "issue_refund",
                        "args": {"invoice_id": "INV-1002"},
                        "id": "fake-call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Refunded $19.99 for INV-1002."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_invoice",
                        "args": {"invoice_id": "INV-1001"},
                        "id": "fake-call-3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="INV-1001: $49.99"),
        ]
    )
    monkeypatch.setattr("swarm_agent.graph.get_chat_model", lambda _route: model)
    return model


@pytest.fixture
def agent_prompts() -> dict[str, str]:
    """Hermetic agent prompts, so tests don't depend on a live MLflow prompt registry."""
    return {
        "triage": "You transfer billing requests to delegate_to_billing.",
        "billing": "You use lookup_invoice/issue_refund to handle billing requests.",
    }
