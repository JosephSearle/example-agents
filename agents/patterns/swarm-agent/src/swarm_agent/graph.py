"""Builds the swarm/handoffs multi-agent graph.

Multiple cooperating agents — Tier 2 in the framework-tiering decision (see
docs/decisions/0001-tech-stack.md). Unlike `supervisor_agent.graph` (also Tier 2, but which
deliberately avoids `langgraph-supervisor` per that library's own soft-deprecation notice), this
pattern's doc (docs/patterns/agent/swarm-handoffs.md) does *not* steer away from
`langgraph-swarm-py` — it's presented as the current LangGraph-ecosystem equivalent of the
handoff pattern, so this module uses it directly: `langgraph_swarm.create_handoff_tool` and
`create_swarm`.

The direct disambiguator from Supervisor: with a handoff, *control moves to the specialist
agent* — it owns the final response, and the swarm remembers which agent was last active so a
follow-up message in the same thread resumes with that same specialist rather than restarting at
`DEFAULT_ACTIVE_AGENT`. Contrast with `supervisor_agent.graph.build_supervisor`, where the
supervisor always keeps ownership of the final reply and sub-agents never persist as "the active
agent" between turns.

Verified against the current `langgraph_swarm` API (not training-data recollection) via the
LangChain reference-docs MCP server: `create_handoff_tool(*, agent_name, name=None,
description=None)` returns a tool whose only parameters are LangGraph-injected (current state
and the tool-call id) — the model never has to supply arguments to trigger a handoff, it just
decides *whether* to call it. `create_swarm(agents, *, default_active_agent, ...)` returns an
uncompiled `StateGraph`; each agent passed to it needs its own `name=` (via `create_agent`'s
`name` parameter), since that name becomes the node the handoff tool's `Command(goto=...)`
targets.
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
from langgraph_swarm import create_handoff_tool, create_swarm

from swarm_agent.tools import BILLING_TOOLS

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from mlflow.entities.model_registry import PromptVersion

__all__ = [
    "AGENTS",
    "DEFAULT_ACTIVE_AGENT",
    "DEFAULT_RECURSION_LIMIT",
    "GATEWAY_ROUTE",
    "PRODUCTION_SCORERS",
    "build_swarm",
    "invoke_config",
    "link_prompts_to_trace",
    "load_agent_prompt",
    "load_agent_prompt_version",
    "prompt_text",
]

# This agent's own MLflow experiment — see agents_common.observability.configure_mlflow.
EXPERIMENT_NAME = "swarm-agent"

# The MLflow AI Gateway route every agent in this module calls — triage and billing alike —
# since this is a reference example rather than a production workload that needs its own
# provisioned model per role.
GATEWAY_ROUTE = "gpt-oss-120b"

# Scorers run continuously against a sampled slice of live production traces — see
# agents_common.observability.register_production_monitors, provisioned via
# packages/mlflow-server/scripts/provision_monitors.py. `owns_handoff_conversation`'s guideline
# text is loaded from packages/mlflow-server/judges/swarm-agent-owns_handoff_conversation.txt,
# the same source that eval suite loads it from — single source of truth, see
# agents_common.judges.
PRODUCTION_SCORERS: list[tuple[Any, float]] = build_production_scorers(
    GATEWAY_ROUTE, [("owns_handoff_conversation", "swarm-agent-owns_handoff_conversation")]
)

# The alias provisioning points at the "live" version of each agent's system prompt — see
# packages/mlflow-server/scripts/provision_prompts.py, which registers this agent's two prompts
# from packages/mlflow-server/prompts/swarm-agent/*.txt.
_PROMPT_ALIAS = PRODUCTION_ALIAS

# The two peer specialists, each with its own registered prompt and its own node name in the
# swarm graph (agent names double as `create_handoff_tool`'s `agent_name` targets).
AGENTS = ("triage", "billing")

# Every new conversation starts here; a follow-up message in the same thread resumes with
# whichever agent last took control instead — see create_swarm's own persistent-active-agent
# behavior, cited in this module's docstring.
DEFAULT_ACTIVE_AGENT = "triage"

# Caps each agent's own internal ReAct loop — same purpose and default as
# react_agent.graph.DEFAULT_RECURSION_LIMIT.
DEFAULT_RECURSION_LIMIT = 25


def load_agent_prompt_version(name: str, *, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch one agent's system prompt version from the MLflow prompt registry.

    Thin wrapper around `agents_common.prompts.load_prompt_version`, binding this agent's
    per-name registry name (`<EXPERIMENT_NAME>-<name>`) and experiment — each of the two peer
    specialists is registered as its own prompt name under this agent's single experiment; see
    provision_prompts.py's per-subdirectory provisioning.

    Returns the full `PromptVersion` (not just its text) so a caller can pass it to
    `link_prompts_to_trace` afterwards — see `swarm_agent.__main__` for the intended usage.

    Args:
        name: One of "triage", "billing".
        alias: Prompt registry alias to load. Defaults to the production alias.
    """
    return load_prompt_version(
        f"{EXPERIMENT_NAME}-{name}", experiment_name=EXPERIMENT_NAME, alias=alias
    )


def load_agent_prompt(name: str, *, alias: str = _PROMPT_ALIAS) -> str:
    """Fetch one agent's system prompt text from the MLflow prompt registry.

    Thin wrapper around `load_agent_prompt_version` for callers that only need the text (e.g.
    `build_swarm`'s default path) and don't need to link the version to a trace afterwards.
    """
    return prompt_text(load_agent_prompt_version(name, alias=alias))


def link_prompts_to_trace(prompt_versions: dict[str, PromptVersion], trace_id: str | None) -> None:
    """Link this invocation's agent prompt versions to a trace.

    Thin wrapper around `agents_common.prompts.link_prompts_to_trace` that accepts this agent's
    name-keyed dict shape — see that function's docstring for `trace_id` semantics.
    """
    _link_prompts_to_trace(list(prompt_versions.values()), trace_id)


def build_swarm(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    agent_prompts: dict[str, str] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the swarm.

    Unlike `supervisor_agent.graph.build_supervisor` (only the top-level supervisor is
    checkpointed, sub-agents are rebuilt fresh per call), the entire swarm — both peer agents —
    is compiled with one checkpointer here: `create_swarm`'s own persistent-active-agent state
    (which agent is currently in control) has to survive across turns in the same thread, so it
    can't be reconstructed fresh on every invocation the way a stateless delegate call can.

    Args:
        checkpointer: A LangGraph checkpointer for the swarm's conversation thread, e.g. a
            `PostgresSaver` (production/integration) or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route every agent calls. Defaults to this package's own
            `GATEWAY_ROUTE`; overridable for tests against a different route.
        agent_prompts: Overrides the registry-fetched system prompts, keyed by agent name
            ("triage", "billing"). Defaults to `None`, which fetches each agent's current
            `production`-aliased prompt via `load_agent_prompt()`. Pass literal strings in tests
            that need a hermetic build with no MLflow prompt-registry dependency.

    Returns:
        A compiled LangGraph graph, invoked with `{"messages": [...]}`.
    """
    prompts = (
        agent_prompts
        if agent_prompts is not None
        else {name: load_agent_prompt(name) for name in AGENTS}
    )

    transfer_to_billing = create_handoff_tool(
        agent_name="billing",
        description="Transfer to the billing specialist for invoice or refund questions.",
    )
    transfer_to_triage = create_handoff_tool(
        agent_name="triage",
        description="Transfer back to triage for anything outside billing.",
    )

    triage_agent = create_agent(
        model=get_chat_model(gateway_route),
        tools=[transfer_to_billing],
        system_prompt=prompts["triage"],
        name="triage",
    )
    billing_agent = create_agent(
        model=get_chat_model(gateway_route),
        tools=[*BILLING_TOOLS, transfer_to_triage],
        system_prompt=prompts["billing"],
        name="billing",
    )

    workflow = create_swarm(
        [triage_agent, billing_agent], default_active_agent=DEFAULT_ACTIVE_AGENT
    )
    return workflow.compile(checkpointer=checkpointer)


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
