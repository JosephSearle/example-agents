"""Builds the map-reduce workflow graph.

This is a workflow, not an agent, per Anthropic's framing (see docs/patterns/agent/map-reduce.md):
which calls run is fixed logic (map every item, then reduce), even though *how many* calls run is
determined at runtime. Built with a raw LangGraph `StateGraph` rather than
`langchain.agents.create_agent` (contrast with `react_agent.graph`), since there's no tool loop
here for `create_agent` to compile.

This is essentially `parallelization_agent.graph.build_sectioning_graph`'s Sectioning shape,
generalized: instead of a fixed set of subtasks written ahead of time (`SECTIONS`), the number of
parallel branches is determined at runtime by graph state, via LangGraph's `Send` API. A routing
function (`continue_to_jokes`) returns one `Send` per input topic — however many that turns out to
be — rather than sectioning's `for section in SECTIONS: graph.add_edge(START, section)` loop over
a name tuple fixed in code. See parallelization_agent.graph's module docstring for the flip side
of this distinction.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from agents_common import get_chat_model
from agents_common.judges import build_production_scorers
from agents_common.prompts import (
    PRODUCTION_ALIAS,
    link_prompts_to_trace,
    load_prompt_version,
    prompt_text,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
import structlog

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from mlflow.entities.model_registry import PromptVersion

_logger = structlog.get_logger(__name__)

__all__ = [
    "GATEWAY_ROUTE",
    "PRODUCTION_SCORERS",
    "JokeState",
    "OverallState",
    "build_map_reduce_graph",
    "invoke_config",
    "link_prompt_to_trace",
    "load_joke_prompt",
    "load_joke_prompt_version",
    "prompt_text",
]

# This agent's own MLflow experiment — see agents_common.observability.configure_mlflow.
EXPERIMENT_NAME = "map-reduce-agent"

# The MLflow AI Gateway route this agent calls — same route react-agent/routing-agent/
# parallelization-agent use, since this is a reference example rather than a production workload
# that needs its own provisioned model.
GATEWAY_ROUTE = "gpt-oss-120b"

# Scorers run continuously against a sampled slice of live production traces — see
# agents_common.observability.register_production_monitors, provisioned via
# packages/mlflow-server/scripts/provision_monitors.py. `relevant_jokes`' guideline text is
# loaded from packages/mlflow-server/judges/map-reduce-agent-relevant_jokes.txt, the same source
# that eval suite loads it from — single source of truth, see agents_common.judges.
PRODUCTION_SCORERS: list[tuple[Any, float]] = build_production_scorers(
    GATEWAY_ROUTE, [("relevant_jokes", "map-reduce-agent-relevant_jokes")]
)

# The alias provisioning points at the "live" version of the worker prompt — see
# packages/mlflow-server/scripts/provision_prompts.py, which registers this agent's single
# prompt from packages/mlflow-server/prompts/map-reduce-agent.txt. Only one prompt (unlike
# parallelization-agent's per-section prompts), since every `generate_joke` worker runs the
# identical instruction against a different topic — mirrors react_agent's single system prompt.
_PROMPT_ALIAS = PRODUCTION_ALIAS


class OverallState(TypedDict):
    """State threaded through the map-reduce graph.

    `jokes` uses an `operator.add` reducer: every dynamically-spawned `generate_joke` worker
    writes to this *same* key, so the reducer is what lets LangGraph accumulate all of their
    outputs into one list — however many workers actually ran — instead of the last writer
    clobbering the rest. Contrast with sectioning's `IncidentState`, where each of the fixed
    three sections writes to its own distinct key and needs no reducer.
    """

    topics: list[str]
    jokes: Annotated[list[str], operator.add]
    summary: str


class JokeState(TypedDict):
    """Per-worker state: exactly what one `Send` call carries into `generate_joke`.

    Deliberately narrower than `OverallState` — a worker only ever sees its own `topic`, never
    the full `topics` list or other workers' state, which is what keeps workers independent (see
    the module docstring's production-checklist note on statelessness).
    """

    topic: str


def load_joke_prompt_version(*, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch this agent's worker prompt version from the MLflow prompt registry.

    Thin wrapper around `agents_common.prompts.load_prompt_version` binding this agent's own
    registry name and experiment. See packages/mlflow-server/scripts/provision_prompts.py, which
    registers `EXPERIMENT_NAME`'s prompt from packages/mlflow-server/prompts/map-reduce-agent.txt.

    Returns the full `PromptVersion` (not just its text) so a caller running the agent can pass
    it to `link_prompt_to_trace` afterwards — see `map_reduce_agent.__main__` for the intended
    usage.
    """
    return load_prompt_version(EXPERIMENT_NAME, experiment_name=EXPERIMENT_NAME, alias=alias)


def load_joke_prompt(*, alias: str = _PROMPT_ALIAS) -> str:
    """Fetch this agent's worker prompt text from the MLflow prompt registry.

    Thin wrapper around `load_joke_prompt_version` for callers that only need the text (e.g.
    `build_map_reduce_graph`'s default path) and don't need to link the version to a trace
    afterwards.
    """
    return prompt_text(load_joke_prompt_version(alias=alias))


def link_prompt_to_trace(prompt_version: PromptVersion, trace_id: str | None) -> None:
    """Link the worker prompt version to a trace so the MLflow UI's trace view shows it.

    Thin wrapper around `agents_common.prompts.link_prompts_to_trace` for this agent's single
    worker prompt — see that function's docstring for `trace_id` semantics.
    """
    link_prompts_to_trace([prompt_version], trace_id)


def build_map_reduce_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    joke_prompt: str | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the map-reduce workflow.

    The caller owns the checkpointer's lifecycle, same convention as
    parallelization_agent.graph.build_sectioning_graph.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration)
            or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route every worker calls. Defaults to this package's own
            `GATEWAY_ROUTE`; overridable for tests against a different route.
        joke_prompt: Overrides the registry-fetched worker prompt. Defaults to `None`, which
            fetches the current `production`-aliased prompt via `load_joke_prompt()`. Pass a
            literal string in tests that need a hermetic build with no MLflow prompt-registry
            dependency.

    Returns:
        A compiled LangGraph graph, invoked with `{"topics": [...], "jokes": [], "summary": ""}`.
        The number of `generate_joke` workers spawned equals `len(topics)` at invoke time — not
        fixed by the graph's structure.
    """
    model = get_chat_model(gateway_route)
    prompt = joke_prompt if joke_prompt is not None else load_joke_prompt()

    def generate_joke(state: JokeState) -> dict[str, list[str]]:
        response = model.invoke(f"{prompt}\n\nTopic: {state['topic']}")
        _logger.info("joke_generated", topic=state["topic"])
        return {"jokes": [str(response.content)]}

    def continue_to_jokes(state: OverallState) -> list[Send]:
        # Runtime-determined fan-out: one Send per topic in state, however many that is — the
        # graph's structure below never names a topic or a worker count.
        _logger.info("fan_out", worker_count=len(state["topics"]))
        return [Send("generate_joke", {"topic": topic}) for topic in state["topics"]]

    def combine_jokes(state: OverallState) -> dict[str, str]:
        # Fixed code rule, not another model call — programmatic aggregation, same "aggregation
        # is programmatic" principle parallelization_agent.graph's aggregate_report follows.
        # Only fires once every Send-spawned generate_joke worker has completed — LangGraph
        # waits for the full dynamic fan-out before this node's inputs are satisfied.
        numbered = "\n".join(f"{i}. {joke}" for i, joke in enumerate(state["jokes"], start=1))
        summary = f"Generated {len(state['jokes'])} joke(s):\n{numbered}"
        _logger.info("jokes_combined", joke_count=len(state["jokes"]))
        return {"summary": summary}

    graph = StateGraph(OverallState)
    graph.add_node("generate_joke", generate_joke)
    graph.add_node("combine_jokes", combine_jokes)

    graph.add_conditional_edges(START, continue_to_jokes, ["generate_joke"])
    graph.add_edge("generate_joke", "combine_jokes")
    graph.add_edge("combine_jokes", END)

    return graph.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str) -> dict[str, Any]:
    """Build the `.invoke()` config for a thread.

    Every call site that runs the compiled graph should route through this, same convention as
    react_agent.graph.invoke_config / parallelization_agent.graph.invoke_config.
    """
    return {"configurable": {"thread_id": thread_id}}
