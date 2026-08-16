"""Builds the routing workflow graph.

This is a workflow, not an agent, per Anthropic's framing (see docs/patterns/agent/routing.md):
the set of possible routes is fixed in code ahead of time — the classifier picks among predefined
categories, it doesn't invent new ones at runtime. Built with a raw LangGraph `StateGraph` rather
than `langchain.agents.create_agent` (contrast with `react_agent.graph`), since there's no tool
loop here for `create_agent` to compile.

Demonstrates the **task routing** flavor from routing.md (as opposed to model-tier routing):
a support ticket is classified into one of three categories, each dispatched to its own
specialized handler prompt via `add_conditional_edges`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

from agents_common import get_chat_model
from agents_common.prompts import (
    PRODUCTION_ALIAS,
    link_prompts_to_trace as _link_prompts_to_trace,
    load_prompt_version,
    prompt_text,
)
from langgraph.graph import END, START, StateGraph
from mlflow.genai.scorers import Guidelines, Safety
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from mlflow.entities.model_registry import PromptVersion

__all__ = [
    "CATEGORIES",
    "PRODUCTION_SCORERS",
    "RouteState",
    "TicketCategory",
    "build_router",
    "invoke_config",
    "link_prompts_to_trace",
    "load_route_prompt",
    "load_route_prompt_version",
    "prompt_text",
]

# This agent's own MLflow experiment — see agents_common.observability.configure_mlflow.
EXPERIMENT_NAME = "routing-agent"

# The MLflow AI Gateway route this agent calls — same route react-agent/prompt-chaining-agent
# use, since this is a reference example rather than a production workload that needs its own
# provisioned model.
GATEWAY_ROUTE = "gpt-oss-120b"

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"

# `PRODUCTION_SCORERS`' judge model — see react_agent.graph's `_MONITOR_JUDGE_MODEL_URI` for why
# `Scorer.start()` requires this `gateway:/<route>` form rather than `_JUDGE_MODEL_URI`'s
# `openai:/<route>`.
_MONITOR_JUDGE_MODEL_URI = f"gateway:/{GATEWAY_ROUTE}"

# Scorers run continuously against a sampled slice of live production traces — see
# agents_common.observability.register_production_monitors, provisioned via
# packages/mlflow-server/scripts/provision_monitors.py. `correct_category` from
# tests/evals/test_quality.py isn't reused here: it's an exact-match check against
# `expectations.expected_category`, which live traces don't have. `relevant_response`'s
# guideline text is reused from that same eval suite.
PRODUCTION_SCORERS: list[tuple[Any, float]] = [
    (
        Guidelines(
            name="relevant_response",
            guidelines=(
                "The response must directly address the customer's ticket and stay "
                "consistent with the category it was routed to (general, refund, or "
                "technical)."
            ),
            model=_MONITOR_JUDGE_MODEL_URI,
        ),
        0.2,
    ),
    (Safety(model=_MONITOR_JUDGE_MODEL_URI), 0.2),  # type: ignore[no-untyped-call]
]

# The alias provisioning points at the "live" version of each route's handler prompt — see
# packages/mlflow-server/scripts/provision_prompts.py, which registers this agent's three
# category prompts from packages/mlflow-server/prompts/routing-agent/*.txt.
_PROMPT_ALIAS = PRODUCTION_ALIAS

CATEGORIES = ("general", "refund", "technical")


class TicketCategory(BaseModel):
    """Structured output for the classifier node — mirrors routing.md's example directly."""

    category: Literal["general", "refund", "technical"] = Field(
        description="The category that best matches this support ticket."
    )


class RouteState(TypedDict):
    """State threaded through the graph.

    `category` is written by the classifier node and read by the conditional edge that dispatches
    to a handler; `response` is written by whichever handler ran.
    """

    message: str
    category: str
    response: str


def load_route_prompt_version(category: str, *, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch one category's handler prompt version from the MLflow prompt registry.

    Thin wrapper around `agents_common.prompts.load_prompt_version`, binding this agent's
    per-category registry name (`<EXPERIMENT_NAME>-<category>`) and experiment — each category is
    registered as its own prompt name under this agent's single experiment; see
    provision_prompts.py's per-subdirectory provisioning.

    Returns the full `PromptVersion` (not just its text) so a caller can pass it to
    `link_prompts_to_trace` afterwards — see `routing_agent.__main__` for the intended usage.

    Args:
        category: One of "general", "refund", "technical".
        alias: Prompt registry alias to load. Defaults to the production alias.
    """
    return load_prompt_version(
        f"{EXPERIMENT_NAME}-{category}", experiment_name=EXPERIMENT_NAME, alias=alias
    )


def load_route_prompt(category: str, *, alias: str = _PROMPT_ALIAS) -> str:
    """Fetch one category's handler prompt text from the MLflow prompt registry.

    Thin wrapper around `load_route_prompt_version` for callers that only need the text (e.g.
    `build_router`'s default path) and don't need to link the version to a trace afterwards.
    """
    return prompt_text(load_route_prompt_version(category, alias=alias))


def link_prompts_to_trace(prompt_versions: dict[str, PromptVersion], trace_id: str | None) -> None:
    """Link this invocation's handler prompt version(s) to a trace.

    Thin wrapper around `agents_common.prompts.link_prompts_to_trace` that accepts this agent's
    category-keyed dict shape — see that function's docstring for `trace_id` semantics. Only the
    prompt versions actually dictionary-supplied (typically just the one category that fired)
    need to be passed in.
    """
    _link_prompts_to_trace(list(prompt_versions.values()), trace_id)


def build_router(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    route_prompts: dict[str, str] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the routing workflow.

    The caller owns the checkpointer's lifecycle, same convention as react_agent.graph.build_agent
    and prompt_chaining_agent.graph.build_chain.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration)
            or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route to call for both classification and handling.
            Defaults to this package's own `GATEWAY_ROUTE`; overridable for tests against a
            different route.
        route_prompts: Overrides the registry-fetched handler prompts, keyed by category
            ("general", "refund", "technical"). Defaults to `None`, which fetches each category's
            current `production`-aliased prompt via `load_route_prompt()`. Pass literal strings in
            tests that need a hermetic build with no MLflow prompt-registry dependency.

    Returns:
        A compiled LangGraph graph, invoked with `{"message": ..., "category": "", "response":
        ""}`.
    """
    model = get_chat_model(gateway_route)
    classifier = model.with_structured_output(TicketCategory)
    prompts = (
        route_prompts
        if route_prompts is not None
        else {category: load_route_prompt(category) for category in CATEGORIES}
    )

    def classify_ticket(state: RouteState) -> dict[str, str]:
        result = classifier.invoke(state["message"])
        return {"category": result.category}  # type: ignore[union-attr]

    def route_from_category(state: RouteState) -> str:
        return state["category"]

    def _make_handler(category: str) -> Any:
        def handle(state: RouteState) -> dict[str, str]:
            response = model.invoke(f"{prompts[category]}\n\n{state['message']}")
            return {"response": str(response.content)}

        return handle

    graph = StateGraph(RouteState)
    graph.add_node("classify_ticket", classify_ticket)
    for category in CATEGORIES:
        graph.add_node(f"handle_{category}", _make_handler(category))

    graph.add_edge(START, "classify_ticket")
    graph.add_conditional_edges(
        "classify_ticket",
        route_from_category,
        {category: f"handle_{category}" for category in CATEGORIES},
    )
    for category in CATEGORIES:
        graph.add_edge(f"handle_{category}", END)

    return graph.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str) -> dict[str, Any]:
    """Build the `.invoke()` config for a thread.

    Every call site that runs the compiled graph should route through this, same convention as
    react_agent.graph.invoke_config / prompt_chaining_agent.graph.invoke_config (no recursion cap
    here — the graph is at most two hops: classify, then dispatch).
    """
    return {"configurable": {"thread_id": thread_id}}
