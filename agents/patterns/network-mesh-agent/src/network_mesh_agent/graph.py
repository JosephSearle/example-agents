"""Builds the network/mesh workflow graph.

Per docs/patterns/agent/network-mesh.md's own honest framing: unlike Supervisor and Swarm, this
topology has no dedicated first-party LangGraph library or actively-maintained doc page — it's
built directly on a raw LangGraph `StateGraph`, following the doc's own worked example (three
peer agents — researcher, critic, writer — each able to route to a different peer, decided at
runtime rather than by a fixed graph shape).

The direct disambiguator from every other multi-agent pattern in this repo: there is no
designated entry-point-owns-the-conversation agent (contrast `swarm_agent`, where a handoff
hands the *entire* rest of the conversation to one specialist) and no single coordinator deciding
every hop (contrast `supervisor_agent`, where the supervisor alone decides who acts next). Here,
`researcher` and `critic` each make their own local routing decision, and that decision can send
control *backward* (critic back to researcher) as well as forward — the "route-back edge" the
topology diagram calls out, and the reason this can't be expressed as `parallelization_agent` or
`orchestrator_workers_agent`'s one-directional fan-out/fan-in shapes.

Rather than a second LLM call per node just to decide where to go next (which the doc's own
`route_after_researcher` example does), each agent's single structured-output call produces its
routing signal *and* its content in one shot — `ResearchFinding.needs_critique`,
`Critique.needs_more_research` — the same "structured output carries the routing decision"
convention `evaluator_optimizer_agent.graph.Evaluation.approved` uses. `route_after_researcher`
and `route_after_critic` below are then pure Python, reading that signal off state, same as
`evaluator_optimizer_agent.graph.route_after_evaluate`.

Per the doc's own practical caveats (O(N²) communication paths, no implicit merge strategy for
shared state), this mesh is deliberately small (three peers) and every field in `MeshState` has
exactly one writer at a time — `messages` is the only accumulated field (`operator.add`, same
convention as every other multi-writer field in this repo), everything else is a plain
last-write-wins field written by one node per turn, so there's no concurrent-write conflict to
resolve.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

from agents_common import get_chat_model, make_prompt_loaders
from agents_common.judges import build_production_scorers
from agents_common.prompts import PRODUCTION_ALIAS, prompt_text
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
import structlog

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from mlflow.entities.model_registry import PromptVersion

_logger = structlog.get_logger(__name__)

__all__ = [
    "AGENTS",
    "DEFAULT_MAX_RESEARCH_ROUNDS",
    "GATEWAY_ROUTE",
    "PRODUCTION_SCORERS",
    "Critique",
    "MeshMessage",
    "MeshState",
    "ResearchFinding",
    "build_mesh_graph",
    "invoke_config",
    "link_prompts_to_trace",
    "load_agent_prompt",
    "load_agent_prompt_version",
    "prompt_text",
]

# This agent's own MLflow experiment — see agents_common.observability.configure_mlflow.
EXPERIMENT_NAME = "network-mesh-agent"

# The MLflow AI Gateway route every peer in this mesh calls — same route every other pattern in
# this repo uses, since this is a reference example rather than a production workload with
# per-agent provisioned tiers.
GATEWAY_ROUTE = "gpt-oss-120b"

# Scorers run continuously against a sampled slice of live production traces — see
# agents_common.observability.register_production_monitors, provisioned via
# packages/mlflow-server/scripts/provision_monitors.py. `grounded_in_research`'s guideline text is
# loaded from packages/mlflow-server/judges/network-mesh-agent-grounded_in_research.txt, the same
# source the eval suite loads it from — single source of truth, see agents_common.judges.
PRODUCTION_SCORERS: list[tuple[Any, float]] = build_production_scorers(
    GATEWAY_ROUTE, [("grounded_in_research", "network-mesh-agent-grounded_in_research")]
)

# The alias provisioning points at the "live" version of each peer's prompt — see
# packages/mlflow-server/scripts/provision_prompts.py, which registers this agent's three peer
# prompts from packages/mlflow-server/prompts/network-mesh-agent/*.txt.
_PROMPT_ALIAS = PRODUCTION_ALIAS

# The three peers in the mesh, each with its own registered prompt and its own node/routing-target
# name — no designated entry-only hub, though `researcher` is where every invocation happens to
# start (see build_mesh_graph's `START` edge).
AGENTS = ("researcher", "critic", "writer")

# Caps the researcher<->critic route-back loop so a mesh that keeps deciding it needs another
# round can't recurse forever — same "cap the loop, force convergence" principle
# evaluator_optimizer_agent.graph.DEFAULT_MAX_ITERATIONS applies to its generate<->evaluate loop.
DEFAULT_MAX_RESEARCH_ROUNDS = 2


class ResearchFinding(BaseModel):
    """Structured output for the researcher node.

    `needs_critique` is the routing signal `route_after_researcher` reads — produced by the same
    call that produces the finding itself, rather than a second LLM call just to decide where to
    go next.
    """

    finding: str = Field(description="What the researcher found for this task.")
    needs_critique: bool = Field(
        description="Whether this finding is thin or contentious enough to need the critic's "
        "review before writing, as opposed to being ready to write up directly."
    )


class Critique(BaseModel):
    """Structured output for the critic node.

    `needs_more_research` is the routing signal `route_after_critic` reads, mirroring
    `ResearchFinding.needs_critique`.
    """

    critique: str = Field(description="The critic's review of the current research finding.")
    needs_more_research: bool = Field(
        description="Whether this critique surfaced gaps serious enough to send the task back to "
        "the researcher for another round, as opposed to being ready to write up as-is."
    )


class MeshMessage(TypedDict):
    """One peer's contribution to the mesh transcript.

    Replaces the earlier `f"[role] text"`-prefixed plain-string convention: peers used to be
    recovered by prefix-matching (`m.startswith("[critic]")`), which is a representation smell —
    this makes the author an explicit, typed field instead of text baked into the string.
    """

    role: str
    content: str


class MeshState(TypedDict):
    """State threaded through the mesh.

    `messages` accumulates every peer's contribution via an `operator.add` reducer — the same
    "every writer, one shared key" shape docs/patterns/agent/network-mesh.md's own `MeshState`
    example uses. `needs_critique`/`needs_more_research`/`research_rounds` are plain
    last-write-wins fields: each has exactly one writer active at a time (the node that just ran),
    so no reducer is needed for them — see this module's docstring on the mesh's merge strategy.
    """

    task: str
    messages: Annotated[list[MeshMessage], operator.add]
    needs_critique: bool
    needs_more_research: bool
    research_rounds: int
    final_answer: str


def load_agent_prompt_version(name: str, *, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch one peer's prompt version from the MLflow prompt registry.

    Thin wrapper around `agents_common.prompts.load_prompt_version`, binding this agent's
    per-name registry name (`<EXPERIMENT_NAME>-<name>`) and experiment — each of the three peers
    is registered as its own prompt name under this agent's single experiment; see
    provision_prompts.py's per-subdirectory provisioning.

    Returns the full `PromptVersion` (not just its text) so a caller can pass it to
    `link_prompts_to_trace` afterwards — see `network_mesh_agent.__main__` for the intended usage.

    Args:
        name: One of "researcher", "critic", "writer".
        alias: Prompt registry alias to load. Defaults to the production alias.
    """
    loaders = make_prompt_loaders(
        f"{EXPERIMENT_NAME}-{name}", experiment_name=EXPERIMENT_NAME, alias=alias
    )
    return loaders.load_version()


def load_agent_prompt(name: str, *, alias: str = _PROMPT_ALIAS) -> str:
    """Fetch one peer's prompt text from the MLflow prompt registry.

    Thin wrapper around `load_agent_prompt_version` for callers that only need the text (e.g.
    `build_mesh_graph`'s default path) and don't need to link the version to a trace afterwards.
    """
    return prompt_text(load_agent_prompt_version(name, alias=alias))


def link_prompts_to_trace(prompt_versions: dict[str, PromptVersion], trace_id: str | None) -> None:
    """Link this invocation's peer prompt versions to a trace.

    Thin wrapper around `agents_common.prompts.link_prompts_to_trace` that accepts this agent's
    name-keyed dict shape — see that function's docstring for `trace_id` semantics.
    """
    loaders = make_prompt_loaders(EXPERIMENT_NAME, experiment_name=EXPERIMENT_NAME)
    for prompt_version in prompt_versions.values():
        loaders.link_to_trace(prompt_version, trace_id)


def build_mesh_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    max_research_rounds: int = DEFAULT_MAX_RESEARCH_ROUNDS,
    agent_prompts: dict[str, str] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the mesh.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration) or
            `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route every peer calls. Defaults to this package's own
            `GATEWAY_ROUTE`; overridable for tests against a different route.
        max_research_rounds: Caps the researcher<->critic route-back loop. Defaults to this
            package's own `DEFAULT_MAX_RESEARCH_ROUNDS`; overridable in tests that need a tighter
            loop bound to assert convergence quickly.
        agent_prompts: Overrides the registry-fetched peer prompts, keyed by agent name
            ("researcher", "critic", "writer"). Defaults to `None`, which fetches each peer's
            current `production`-aliased prompt via `load_agent_prompt()`. Pass literal strings in
            tests that need a hermetic build with no MLflow prompt-registry dependency.

    Returns:
        A compiled LangGraph graph, invoked with `{"task": ..., "messages": [],
        "needs_critique": False, "needs_more_research": False, "research_rounds": 0,
        "final_answer": ""}`.
    """
    model = get_chat_model(gateway_route)
    prompts = (
        agent_prompts
        if agent_prompts is not None
        else {name: load_agent_prompt(name) for name in AGENTS}
    )

    def researcher(state: MeshState) -> dict[str, Any]:
        prior_critique = next(
            (m["content"] for m in reversed(state["messages"]) if m["role"] == "critic"), None
        )
        context = f"\n\nPrior critique to address:\n{prior_critique}" if prior_critique else ""
        # Bound fresh per call (not once at graph-build time): across a researcher<->critic loop
        # this node runs more than once per invocation, and structured-output binding is cheap
        # and stateless to redo — same reason `critic` below rebinds too.
        result = model.with_structured_output(ResearchFinding).invoke(
            f"{prompts['researcher']}\n\nTask: {state['task']}{context}"
        )
        assert isinstance(result, ResearchFinding)
        research_rounds = state["research_rounds"] + 1
        _logger.info(
            "researched", research_rounds=research_rounds, needs_critique=result.needs_critique
        )
        return {
            "messages": [{"role": "researcher", "content": result.finding}],
            "needs_critique": result.needs_critique,
            "research_rounds": research_rounds,
        }

    def route_after_researcher(state: MeshState) -> Literal["critic", "writer"]:
        decision: Literal["critic", "writer"]
        if state["research_rounds"] >= max_research_rounds:
            decision = "writer"
        else:
            decision = "critic" if state["needs_critique"] else "writer"
        _logger.info("mesh_routed", from_node="researcher", to=decision)
        return decision

    def critic(state: MeshState) -> dict[str, Any]:
        last_finding = next(
            m["content"] for m in reversed(state["messages"]) if m["role"] == "researcher"
        )
        result = model.with_structured_output(Critique).invoke(
            f"{prompts['critic']}\n\nTask: {state['task']}\n\n{last_finding}"
        )
        assert isinstance(result, Critique)
        _logger.info("critiqued", needs_more_research=result.needs_more_research)
        return {
            "messages": [{"role": "critic", "content": result.critique}],
            "needs_more_research": result.needs_more_research,
        }

    def route_after_critic(state: MeshState) -> Literal["researcher", "writer"]:
        decision: Literal["researcher", "writer"]
        if state["research_rounds"] >= max_research_rounds:
            decision = "writer"
        else:
            decision = "researcher" if state["needs_more_research"] else "writer"
        _logger.info(
            "mesh_routed",
            from_node="critic",
            to=decision,
            research_rounds=state["research_rounds"],
        )
        return decision

    def writer(state: MeshState) -> dict[str, str | list[MeshMessage]]:
        transcript = "\n\n".join(f"[{m['role']}] {m['content']}" for m in state["messages"])
        response = model.invoke(f"{prompts['writer']}\n\nTask: {state['task']}\n\n{transcript}")
        answer = str(response.content)
        _logger.info("written", transcript_message_count=len(state["messages"]))
        return {
            "messages": [{"role": "writer", "content": answer}],
            "final_answer": answer,
        }

    graph = StateGraph(MeshState)
    graph.add_node("researcher", researcher)
    graph.add_node("critic", critic)
    graph.add_node("writer", writer)

    graph.add_edge(START, "researcher")
    graph.add_conditional_edges("researcher", route_after_researcher, ["critic", "writer"])
    graph.add_conditional_edges("critic", route_after_critic, ["researcher", "writer"])
    graph.add_edge("writer", END)

    return graph.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str) -> dict[str, Any]:
    """Build the `.invoke()` config for a thread.

    Every call site that runs the compiled graph should route through this, same convention as
    react_agent.graph.invoke_config / orchestrator_workers_agent.graph.invoke_config.
    """
    return {"configurable": {"thread_id": thread_id}}
