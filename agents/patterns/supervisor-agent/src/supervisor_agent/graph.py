"""Builds the supervisor multi-agent graph.

Multiple cooperating agents — Tier 2 in the framework-tiering decision (see
docs/decisions/0001-tech-stack.md), since one `create_agent` ReAct loop can't express "delegate
to a specialist sub-agent, then synthesize its result" the way a single tool call can.

Deliberately does **not** depend on the `langgraph-supervisor` package the ADR's tier table
names as an example — see docs/patterns/agent/supervisor.md's own tooling note: that library's
README now says *"We now recommend using the supervisor pattern directly via tools rather than
this library for most use cases. The tool-calling approach gives you more control over context
engineering and is the recommended pattern in the LangChain multi-agent guide."* This module
follows that current guidance instead: each sub-agent is a full `create_agent()` instance in its
own right, wrapped as an `@tool`-decorated `delegate_to_*` function, and the supervisor is
itself another `create_agent()` whose only tools are those delegate wrappers — no raw
`StateGraph`, no `langgraph-supervisor` dependency, just `langchain.agents.create_agent`
composed twice over. Contrast with `react_agent.graph.build_agent` (Tier 1): a single agent with
its own tools; here, the "tools" the top-level agent sees are themselves full agents.

Sub-agents never see the user directly, and never see each other — the supervisor is the only
thing a caller talks to, and it only ever sees each sub-agent's *final* output, not its internal
reasoning. That's the boundary against handoffs/swarm (see docs/patterns/agent/swarm-handoffs.md,
not implemented in this repo): there, a user ends up talking directly to whichever specialist
took over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents_common import get_chat_model
from agents_common.judges import build_production_scorers
from agents_common.prompts import (
    PRODUCTION_ALIAS,
    link_prompts_to_trace as _link_prompts_to_trace,
    load_prompt_version,
    prompt_text,
)
from langchain.agents import create_agent
from langchain_core.tools import tool
import structlog

from supervisor_agent.tools import MATH_TOOLS, TEXT_TOOLS

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from mlflow.entities.model_registry import PromptVersion

_logger = structlog.get_logger(__name__)

__all__ = [
    "AGENTS",
    "DEFAULT_RECURSION_LIMIT",
    "GATEWAY_ROUTE",
    "PRODUCTION_SCORERS",
    "build_supervisor",
    "invoke_config",
    "link_prompts_to_trace",
    "load_agent_prompt",
    "load_agent_prompt_version",
    "prompt_text",
]

# This agent's own MLflow experiment — see agents_common.observability.configure_mlflow.
EXPERIMENT_NAME = "supervisor-agent"

# The MLflow AI Gateway route every agent in this module calls — supervisor and both sub-agents
# alike — since this is a reference example rather than a production workload that needs its
# own provisioned model per role.
GATEWAY_ROUTE = "gpt-oss-120b"

# Scorers run continuously against a sampled slice of live production traces — see
# agents_common.observability.register_production_monitors, provisioned via
# packages/mlflow-server/scripts/provision_monitors.py. `delegates_appropriately`'s guideline
# text is loaded from
# packages/mlflow-server/judges/supervisor-agent-delegates_appropriately.txt, the same source
# that eval suite loads it from — single source of truth, see agents_common.judges.
PRODUCTION_SCORERS: list[tuple[Any, float]] = build_production_scorers(
    GATEWAY_ROUTE, [("delegates_appropriately", "supervisor-agent-delegates_appropriately")]
)

# The alias provisioning points at the "live" version of each agent's system prompt — see
# packages/mlflow-server/scripts/provision_prompts.py, which registers this agent's three
# prompts from packages/mlflow-server/prompts/supervisor-agent/*.txt.
_PROMPT_ALIAS = PRODUCTION_ALIAS

# The supervisor and its two sub-agents, each with its own registered prompt — not a "step"
# sequence like prompt-chaining or evaluator-optimizer's STEPS, but the naming convention
# (`f"{EXPERIMENT_NAME}-{name}"` per registered prompt) is the same.
AGENTS = ("supervisor", "math", "text")

# Caps each agent's own internal ReAct loop (supervisor and both sub-agents each run one) — same
# purpose and default as react_agent.graph.DEFAULT_RECURSION_LIMIT.
DEFAULT_RECURSION_LIMIT = 25


def load_agent_prompt_version(name: str, *, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch one agent's system prompt version from the MLflow prompt registry.

    Thin wrapper around `agents_common.prompts.load_prompt_version`, binding this agent's
    per-name registry name (`<EXPERIMENT_NAME>-<name>`) and experiment — each of the supervisor
    and its two sub-agents is registered as its own prompt name under this agent's single
    experiment; see provision_prompts.py's per-subdirectory provisioning.

    Returns the full `PromptVersion` (not just its text) so a caller can pass it to
    `link_prompts_to_trace` afterwards — see `supervisor_agent.__main__` for the intended usage.

    Args:
        name: One of "supervisor", "math", "text".
        alias: Prompt registry alias to load. Defaults to the production alias.
    """
    return load_prompt_version(
        f"{EXPERIMENT_NAME}-{name}", experiment_name=EXPERIMENT_NAME, alias=alias
    )


def load_agent_prompt(name: str, *, alias: str = _PROMPT_ALIAS) -> str:
    """Fetch one agent's system prompt text from the MLflow prompt registry.

    Thin wrapper around `load_agent_prompt_version` for callers that only need the text (e.g.
    `build_supervisor`'s default path) and don't need to link the version to a trace afterwards.
    """
    return prompt_text(load_agent_prompt_version(name, alias=alias))


def link_prompts_to_trace(prompt_versions: dict[str, PromptVersion], trace_id: str | None) -> None:
    """Link this invocation's agent prompt versions to a trace.

    Thin wrapper around `agents_common.prompts.link_prompts_to_trace` that accepts this agent's
    name-keyed dict shape — see that function's docstring for `trace_id` semantics.
    """
    _link_prompts_to_trace(list(prompt_versions.values()), trace_id)


def build_supervisor(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    agent_prompts: dict[str, str] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the supervisor and its two sub-agents.

    Only the supervisor itself is checkpointed — the caller owns that checkpointer's lifecycle,
    same convention as react_agent.graph.build_agent. The two sub-agents are rebuilt fresh on
    every call and never see a checkpointer: each `delegate_to_*` call is a single, stateless
    turn from the sub-agent's point of view (it never needs to remember a prior delegation),
    matching docs/patterns/agent/supervisor.md's own worked example.

    Args:
        checkpointer: A LangGraph checkpointer for the supervisor's own conversation thread,
            e.g. a `PostgresSaver` (production/integration) or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route every agent calls. Defaults to this package's own
            `GATEWAY_ROUTE`; overridable for tests against a different route.
        agent_prompts: Overrides the registry-fetched system prompts, keyed by agent name
            ("supervisor", "math", "text"). Defaults to `None`, which fetches each agent's
            current `production`-aliased prompt via `load_agent_prompt()`. Pass literal strings
            in tests that need a hermetic build with no MLflow prompt-registry dependency.

    Returns:
        A compiled LangGraph graph (the supervisor's own `create_agent` graph), invoked with
        `{"messages": [...]}`.
    """
    prompts = (
        agent_prompts
        if agent_prompts is not None
        else {name: load_agent_prompt(name) for name in AGENTS}
    )

    math_agent = create_agent(
        model=get_chat_model(gateway_route),
        tools=MATH_TOOLS,
        system_prompt=prompts["math"],
    )
    text_agent = create_agent(
        model=get_chat_model(gateway_route),
        tools=TEXT_TOOLS,
        system_prompt=prompts["text"],
    )

    @tool
    def delegate_to_math(request: str) -> str:
        """Delegate an arithmetic/calculation request to the math sub-agent."""
        _logger.info("delegated", to="math")
        result = math_agent.invoke({"messages": [{"role": "user", "content": request}]})
        return str(result["messages"][-1].content)

    @tool
    def delegate_to_text(request: str) -> str:
        """Delegate a word-counting or text-reversal request to the text sub-agent."""
        _logger.info("delegated", to="text")
        result = text_agent.invoke({"messages": [{"role": "user", "content": request}]})
        return str(result["messages"][-1].content)

    return create_agent(
        model=get_chat_model(gateway_route),
        tools=[delegate_to_math, delegate_to_text],
        system_prompt=prompts["supervisor"],
        checkpointer=checkpointer,
    )


def invoke_config(
    thread_id: str,
    *,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> RunnableConfig:
    """Build the `.invoke()` config for a thread, with the recursion cap applied.

    Every call site that runs the compiled graph should route through this, same convention as
    react_agent.graph.invoke_config.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
