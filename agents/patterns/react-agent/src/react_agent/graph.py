"""Builds the ReAct agent graph.

`create_agent` (LangChain v1) compiles down to a LangGraph graph itself, so everything that
applies to "real" LangGraph graphs elsewhere in this repo — checkpointers, stores, streaming,
structured `interrupt()` — applies here too. This is tier 1 in the framework-tiering decision;
see docs/decisions/0001-tech-stack.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents_common import get_chat_model
from agents_common.prompts import (
    PRODUCTION_ALIAS,
    link_prompts_to_trace,
    load_prompt_version,
    prompt_text,
)
from langchain.agents import create_agent
from mlflow.genai.scorers import Guidelines, Safety
from pydantic import BaseModel, Field, ValidationError

from react_agent.tools import TOOLS

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from mlflow.entities.model_registry import PromptVersion

__all__ = [
    "PRODUCTION_SCORERS",
    "AgentResponse",
    "build_agent",
    "extract_response",
    "invoke_config",
    "link_prompt_to_trace",
    "load_system_prompt",
    "load_system_prompt_version",
    "prompt_text",
]

# This agent's own MLflow experiment — see agents_common.observability.configure_mlflow for
# why this is a per-agent constant rather than a shared name pulled from .env.
EXPERIMENT_NAME = "react-agent"

# The MLflow AI Gateway route this agent calls — provisioned via
# packages/mlflow-server/scripts/provision_gateway_route.py against our self-hosted
# OpenAI-compatible model. See agents_common.models.get_chat_model.
GATEWAY_ROUTE = "gpt-oss-120b"

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"

# `PRODUCTION_SCORERS`' judge model, unlike `_JUDGE_MODEL_URI` above: `Scorer.start()` (MLflow's
# production-monitoring API) rejects any scorer whose `model=` isn't in `gateway:/<route>` form —
# it validates server-side that the model resolves to a real MLflow AI Gateway endpoint, unlike
# `mlflow.genai.evaluate()`'s judge calls, which are fine with `openai:/<route>` because those
# are routed through the OPENAI_API_BASE env var trick in tests/evals/test_quality.py instead of
# MLflow's own gateway integration. Confirmed against a live mlflow-server: `openai:/...` here
# fails `.start()` with "does not use a gateway model".
_MONITOR_JUDGE_MODEL_URI = f"gateway:/{GATEWAY_ROUTE}"

# Scorers run continuously against a sampled slice of live production traces — see
# agents_common.observability.register_production_monitors, provisioned via
# packages/mlflow-server/scripts/provision_monitors.py. Distinct from the eval-set scorers in
# tests/evals/test_quality.py: no `Correctness` here, since production questions have no
# ground-truth `expected_facts` to judge against; `concise_answer`'s guideline text is reused
# from that same eval suite. Sampled at 0.2 rather than every trace, to bound judge-call cost
# against real traffic volume.
PRODUCTION_SCORERS: list[tuple[Any, float]] = [
    (
        Guidelines(
            name="concise_answer",
            guidelines="The answer must be a direct response with no meta-commentary about tool usage.",
            model=_MONITOR_JUDGE_MODEL_URI,
        ),
        0.2,
    ),
    (Safety(model=_MONITOR_JUDGE_MODEL_URI), 0.2),  # type: ignore[no-untyped-call]
]

# The ReAct loop's length isn't fixed in code — the model decides how many Thought/Action/
# Observation cycles it needs. Without a cap, a confused model (or a broken tool feeding it
# unhelpful observations) can loop indefinitely. LangGraph enforces this as a step count on the
# compiled graph, set per-invocation via `config`, not on `create_agent` itself — see
# `invoke_config()` below.
DEFAULT_RECURSION_LIMIT = 25

# The alias provisioning points at the "live" version of a prompt — see
# packages/mlflow-server/scripts/provision_prompts.py, which registers this agent's system
# prompt from packages/mlflow-server/prompts/react-agent.txt and aliases it here.
_PROMPT_ALIAS = PRODUCTION_ALIAS


class AgentResponse(BaseModel):
    """Structured final response the agent must produce."""

    answer: str = Field(description="The direct answer to the user's question.")
    used_tools: list[str] = Field(
        default_factory=list,
        description="Names of the tools that were called while producing this answer.",
    )


def load_system_prompt_version(*, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch this agent's system prompt version from the MLflow prompt registry.

    Thin wrapper around `agents_common.prompts.load_prompt_version` binding this agent's own
    registry name and experiment. See packages/mlflow-server/scripts/provision_prompts.py, which
    registers `EXPERIMENT_NAME`'s prompt from packages/mlflow-server/prompts/react-agent.txt and
    points `alias` at it — that script must have run (or `make up`, which now runs it
    automatically) before this can succeed.

    Returns the full `PromptVersion` (not just its text) so a caller running the agent can pass
    it to `link_prompt_to_trace` afterwards — see `react_agent.__main__` for the intended usage.
    """
    return load_prompt_version(EXPERIMENT_NAME, experiment_name=EXPERIMENT_NAME, alias=alias)


def load_system_prompt(*, alias: str = _PROMPT_ALIAS) -> str:
    """Fetch this agent's system prompt text from the MLflow prompt registry.

    Thin wrapper around `load_system_prompt_version` for callers that only need the text (e.g.
    `build_agent`'s default path) and don't need to link the version to a trace afterwards.
    """
    return prompt_text(load_system_prompt_version(alias=alias))


def link_prompt_to_trace(prompt_version: PromptVersion, trace_id: str | None) -> None:
    """Link a prompt version to a trace so the MLflow UI's trace view shows it under "Prompts".

    Thin wrapper around `agents_common.prompts.link_prompts_to_trace` for this agent's single
    system prompt — see that function's docstring for `trace_id` semantics.
    """
    link_prompts_to_trace([prompt_version], trace_id)


def build_agent(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    system_prompt: str | None = None,
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
        system_prompt: Overrides the registry-fetched prompt. Defaults to `None`, which fetches
            the current `production`-aliased prompt via `load_system_prompt()` — the normal
            runtime path. Pass a literal string in tests that need a hermetic build with no
            MLflow prompt-registry dependency.

    Returns:
        A compiled LangGraph graph, invoked with `{"messages": [...]}`.
    """
    model = get_chat_model(gateway_route)
    prompt = system_prompt if system_prompt is not None else load_system_prompt()

    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=prompt,
        response_format=AgentResponse,
        checkpointer=checkpointer,
    )


def invoke_config(
    thread_id: str,
    *,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> RunnableConfig:
    """Build the `.invoke()` config for a thread, with the recursion cap applied.

    Every call site that runs the compiled graph should route through this so the cap in
    `DEFAULT_RECURSION_LIMIT` is actually enforced rather than silently left unbounded.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }


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
