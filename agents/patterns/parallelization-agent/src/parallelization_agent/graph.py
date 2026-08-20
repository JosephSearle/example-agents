"""Builds the parallelization workflow graphs.

This is a workflow, not an agent, per Anthropic's framing (see
docs/patterns/agent/parallelization.md): which calls run and how their outputs are combined is
fixed in code ahead of time. Built with a raw LangGraph `StateGraph` rather than
`langchain.agents.create_agent` (contrast with `react_agent.graph`), since there's no tool loop
here for `create_agent` to compile.

Demonstrates both flavors from parallelization.md:

- **Sectioning** (`build_sectioning_graph`) — an incident report is split into three independent
  subtasks (`summarize`, `assess_severity`, `extract_action_items`) that all run concurrently off
  `START` via plain static edges, fanning into a code-only `aggregate_report` node. This is the
  shape `routing_agent.graph.build_router`'s `add_conditional_edges` can't express: routing picks
  *one* branch per invocation, sectioning runs *all* of them. Static edges are used rather than
  the `Send` API deliberately: the three sections are fixed in code, not determined at runtime —
  `Send`-based fan-out is reserved for the runtime-determined case (see map-reduce-agent). Each
  section's instruction prompt is fetched from the MLflow prompt registry the same way
  `routing_agent` fetches its per-category handler prompts (one prompt name per section:
  `parallelization-agent-summarize`, `parallelization-agent-assess_severity`,
  `parallelization-agent-extract_action_items`).
- **Voting** (`build_voting_graph`) — the same prompt is run `n` times in parallel (`vote_1`,
  `vote_2`, `vote_3` by default), and a deterministic `aggregate_votes` node picks the majority
  verdict. Kept judge-free (no second LLM call to aggregate) to keep the reference example's
  nondeterminism contained to the voting calls themselves.
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
    "GATEWAY_ROUTE",
    "PRODUCTION_SCORERS",
    "SECTIONS",
    "IncidentState",
    "SeverityAssessment",
    "VoteState",
    "build_sectioning_graph",
    "build_voting_graph",
    "invoke_config",
    "link_prompts_to_trace",
    "load_section_prompt",
    "load_section_prompt_version",
    "prompt_text",
]

# This agent's own MLflow experiment — see agents_common.observability.configure_mlflow.
EXPERIMENT_NAME = "parallelization-agent"

# The MLflow AI Gateway route this agent calls — same route react-agent/routing-agent use, since
# this is a reference example rather than a production workload that needs its own provisioned
# model.
GATEWAY_ROUTE = "gpt-oss-120b"

# Scorers run continuously against a sampled slice of live production traces — see
# agents_common.observability.register_production_monitors, provisioned via
# packages/mlflow-server/scripts/provision_monitors.py. `coherent_report`'s guideline text is
# loaded from packages/mlflow-server/judges/parallelization-agent-coherent_report.txt, the same
# source that eval suite loads it from — single source of truth, see agents_common.judges.
PRODUCTION_SCORERS: list[tuple[Any, float]] = build_production_scorers(
    GATEWAY_ROUTE, [("coherent_report", "parallelization-agent-coherent_report")]
)

# The alias provisioning points at the "live" version of each section's instruction prompt — see
# packages/mlflow-server/scripts/provision_prompts.py, which registers this agent's three
# section prompts from packages/mlflow-server/prompts/parallelization-agent/*.txt.
_PROMPT_ALIAS = PRODUCTION_ALIAS

SECTIONS = ("summarize", "assess_severity", "extract_action_items")


class SeverityAssessment(BaseModel):
    """Structured output for the `assess_severity` section."""

    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="The severity that best matches this incident."
    )


class ActionItems(BaseModel):
    """Structured output for the `extract_action_items` section."""

    items: list[str] = Field(description="Concrete follow-up actions for this incident.")


class IncidentState(TypedDict):
    """State threaded through the sectioning graph.

    `summary`, `severity`, and `action_items` are each written by exactly one of the three
    parallel section nodes — since they're distinct keys, LangGraph can merge all three nodes'
    updates for the same superstep without a concurrent-write conflict, no `Annotated[...,
    operator.add]` reducer needed. `report` is written last, by the fan-in `aggregate_report`
    node.
    """

    incident_text: str
    summary: str
    severity: str
    action_items: list[str]
    report: str


class VoteState(TypedDict):
    """State threaded through the voting graph.

    `attempts` uses an `operator.add` reducer: unlike sectioning's distinct per-node keys, every
    `vote_n` node writes to the *same* key, so the reducer is what lets LangGraph accumulate all
    of them into one list instead of the last writer clobbering the rest.
    """

    prompt: str
    attempts: Annotated[list[str], operator.add]
    verdict: str


# One `PromptLoaders` bundle per section, each bound to this agent's per-section registry name
# (`<EXPERIMENT_NAME>-<section>`) and the production alias — see provision_prompts.py's
# per-subdirectory provisioning.
_section_loaders = {
    section: make_prompt_loaders(
        f"{EXPERIMENT_NAME}-{section}", experiment_name=EXPERIMENT_NAME, alias=_PROMPT_ALIAS
    )
    for section in SECTIONS
}


def load_section_prompt_version(section: str, *, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch one section's instruction prompt version from the MLflow prompt registry.

    Thin wrapper around `agents_common.make_prompt_loaders`, binding this agent's per-section
    registry name (`<EXPERIMENT_NAME>-<section>`) and experiment — each section is registered as
    its own prompt name under this agent's single experiment; see provision_prompts.py's
    per-subdirectory provisioning.

    Returns the full `PromptVersion` (not just its text) so a caller can pass it to
    `link_prompts_to_trace` afterwards — see `parallelization_agent.__main__` for the intended
    usage.

    Args:
        section: One of "summarize", "assess_severity", "extract_action_items".
        alias: Prompt registry alias to load. Defaults to the production alias.
    """
    if alias == _PROMPT_ALIAS:
        return _section_loaders[section].load_version()
    return make_prompt_loaders(
        f"{EXPERIMENT_NAME}-{section}", experiment_name=EXPERIMENT_NAME, alias=alias
    ).load_version()


def load_section_prompt(section: str, *, alias: str = _PROMPT_ALIAS) -> str:
    """Fetch one section's instruction prompt text from the MLflow prompt registry.

    Thin wrapper around `load_section_prompt_version` for callers that only need the text (e.g.
    `build_sectioning_graph`'s default path) and don't need to link the version to a trace
    afterwards.
    """
    return prompt_text(load_section_prompt_version(section, alias=alias))


def link_prompts_to_trace(prompt_versions: dict[str, PromptVersion], trace_id: str | None) -> None:
    """Link this invocation's section prompt version(s) to a trace.

    Thin wrapper around `agents_common.make_prompt_loaders`'s per-section `link_to_trace` that
    accepts this agent's section-keyed dict shape — see that function's docstring for `trace_id`
    semantics.
    """
    for section, prompt_version in prompt_versions.items():
        _section_loaders[section].link_to_trace(prompt_version, trace_id)


def build_sectioning_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    section_prompts: dict[str, str] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the sectioning workflow.

    The caller owns the checkpointer's lifecycle, same convention as
    routing_agent.graph.build_router.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration)
            or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route to call for all three sections. Defaults to this
            package's own `GATEWAY_ROUTE`; overridable for tests against a different route.
        section_prompts: Overrides the registry-fetched instruction prompts, keyed by section
            ("summarize", "assess_severity", "extract_action_items"). Defaults to `None`, which
            fetches each section's current `production`-aliased prompt via `load_section_prompt()`.
            Pass literal strings in tests that need a hermetic build with no MLflow prompt-registry
            dependency.

    Returns:
        A compiled LangGraph graph, invoked with `{"incident_text": ..., "summary": "",
        "severity": "", "action_items": [], "report": ""}`.
    """
    model = get_chat_model(gateway_route)
    severity_model = model.with_structured_output(SeverityAssessment)
    action_items_model = model.with_structured_output(ActionItems)
    prompts = (
        section_prompts
        if section_prompts is not None
        else {section: load_section_prompt(section) for section in SECTIONS}
    )

    def summarize(state: IncidentState) -> dict[str, str]:
        response = model.invoke(f"{prompts['summarize']}\n\n{state['incident_text']}")
        _logger.info("section_completed", section="summarize")
        return {"summary": str(response.content)}

    def assess_severity(state: IncidentState) -> dict[str, str]:
        result = severity_model.invoke(f"{prompts['assess_severity']}\n\n{state['incident_text']}")
        _logger.info("section_completed", section="assess_severity", severity=result.severity)  # type: ignore[union-attr]
        return {"severity": result.severity}  # type: ignore[union-attr]

    def extract_action_items(state: IncidentState) -> dict[str, list[str]]:
        result = action_items_model.invoke(
            f"{prompts['extract_action_items']}\n\n{state['incident_text']}"
        )
        assert isinstance(result, ActionItems)
        _logger.info(
            "section_completed", section="extract_action_items", item_count=len(result.items)
        )
        return {"action_items": result.items}

    def aggregate_report(state: IncidentState) -> dict[str, str]:
        # Fixed code rule, not another model call — programmatic aggregation, per
        # parallelization.md's "Aggregation is programmatic" principle.
        items = "\n".join(f"- {item}" for item in state["action_items"])
        report = (
            f"Severity: {state['severity']}\n\n"
            f"Summary: {state['summary']}\n\n"
            f"Action items:\n{items}"
        )
        _logger.info("report_aggregated", severity=state["severity"])
        return {"report": report}

    graph = StateGraph(IncidentState)
    graph.add_node("summarize", summarize)
    graph.add_node("assess_severity", assess_severity)
    graph.add_node("extract_action_items", extract_action_items)
    graph.add_node("aggregate_report", aggregate_report)

    for section in SECTIONS:
        graph.add_edge(START, section)
        graph.add_edge(section, "aggregate_report")
    graph.add_edge("aggregate_report", END)

    return graph.compile(checkpointer=checkpointer)


def build_voting_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    n: int = 3,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the voting workflow.

    `n` statically-defined voter nodes each call the model once against the same prompt and
    append their attempt; `aggregate_votes` picks the majority verdict deterministically. `n` is
    fixed at graph-build time (not runtime-determined), matching sectioning's static-edges
    convention — see the module docstring for why this isn't `Send`-based.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration)
            or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route to call for each voter. Defaults to this package's
            own `GATEWAY_ROUTE`; overridable for tests against a different route.
        n: Number of parallel voters. Defaults to 3.

    Returns:
        A compiled LangGraph graph, invoked with `{"prompt": ..., "attempts": [], "verdict": ""}`.
    """
    model = get_chat_model(gateway_route)

    # Every voter node runs identical logic (call the model once on the same prompt, append the
    # attempt) — registered under `n` distinct node names below rather than built from a factory,
    # since a factory returning `Callable[[VoteState], ...]` loses the literal node-function type
    # `StateGraph.add_node` expects.
    def vote(state: VoteState) -> dict[str, list[str]]:
        response = model.invoke(state["prompt"])
        return {"attempts": [str(response.content)]}

    def aggregate_votes(state: VoteState) -> dict[str, str]:
        # Deterministic majority vote — no second LLM-judge call, so this graph's only
        # nondeterminism is in the `n` voter calls themselves.
        counts: dict[str, int] = {}
        for attempt in state["attempts"]:
            counts[attempt] = counts.get(attempt, 0) + 1
        verdict = max(counts, key=lambda attempt: counts[attempt])
        _logger.info("votes_aggregated", attempt_count=len(state["attempts"]), verdict=verdict)
        return {"verdict": verdict}

    graph = StateGraph(VoteState)
    voter_names = [f"vote_{i}" for i in range(1, n + 1)]
    for name in voter_names:
        graph.add_node(name, vote)
    graph.add_node("aggregate_votes", aggregate_votes)

    for name in voter_names:
        graph.add_edge(START, name)
        graph.add_edge(name, "aggregate_votes")
    graph.add_edge("aggregate_votes", END)

    return graph.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str) -> dict[str, Any]:
    """Build the `.invoke()` config for a thread.

    Every call site that runs a compiled graph should route through this, same convention as
    react_agent.graph.invoke_config / routing_agent.graph.invoke_config.
    """
    return {"configurable": {"thread_id": thread_id}}
