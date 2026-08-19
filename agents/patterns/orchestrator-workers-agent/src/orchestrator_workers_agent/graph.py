"""Builds the orchestrator-workers workflow graph.

This is a boundary case in Anthropic's own taxonomy (see
docs/patterns/agent/orchestrator-workers.md): orchestrate → workers → synthesize is a fixed
high-level shape, so it's still a workflow — but unlike every other workflow pattern in this
repo, the *number and nature* of the middle stage's subtasks is decided by an LLM (the
orchestrator) at runtime, not fixed in code. Built with a raw LangGraph `StateGraph` rather than
`langchain.agents.create_agent` (contrast with `react_agent.graph`), since there's no tool loop
here for `create_agent` to compile.

Contrast with the two other dynamic-fan-out-shaped patterns in this repo:
- `parallelization_agent.graph.build_sectioning_graph` fixes both *which* subtasks run and *how
  many* — three named sections, always.
- `map_reduce_agent.graph.build_map_reduce_graph` fixes *what* each worker does (one fixed
  prompt) but lets *how many* run vary with input length (one `Send` per list item).
- This graph lets an orchestrator LLM call decide *both* — how many subtasks, and what each one
  actually needs to accomplish — via structured output (`TaskBreakdown`), before the same
  `Send`-based fan-out mechanism map-reduce uses spawns one worker per decided subtask.

The docs page recommends model-tiering (a stronger model for orchestration/synthesis, a cheaper
model for workers) as a cost lever. This reference implementation reuses one `GATEWAY_ROUTE` for
all three stages, same as every other pattern in this repo (a single self-hosted reference model,
not a production workload with multiple provisioned tiers) — a real deployment following this
pattern would pass a second, cheaper `gateway_route` into the worker call.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from agents_common import get_chat_model
from agents_common.judges import build_production_scorers
from agents_common.prompts import (
    PRODUCTION_ALIAS,
    link_prompts_to_trace as _link_prompts_to_trace,
    load_prompt_version,
    prompt_text,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
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
    "STEPS",
    "OverallState",
    "Subtask",
    "TaskBreakdown",
    "WorkerState",
    "build_orchestrator_graph",
    "invoke_config",
    "link_prompts_to_trace",
    "load_step_prompt",
    "load_step_prompt_version",
    "prompt_text",
]

# This agent's own MLflow experiment — see agents_common.observability.configure_mlflow.
EXPERIMENT_NAME = "orchestrator-workers-agent"

# The MLflow AI Gateway route this agent calls — same route react-agent/parallelization-agent/
# map-reduce-agent use, since this is a reference example rather than a production workload that
# needs its own provisioned model.
GATEWAY_ROUTE = "gpt-oss-120b"

# Scorers run continuously against a sampled slice of live production traces — see
# agents_common.observability.register_production_monitors, provisioned via
# packages/mlflow-server/scripts/provision_monitors.py. `coherent_synthesis`'s guideline text is
# loaded from packages/mlflow-server/judges/orchestrator-workers-agent-coherent_synthesis.txt,
# the same source that eval suite loads it from — single source of truth, see
# agents_common.judges.
PRODUCTION_SCORERS: list[tuple[Any, float]] = build_production_scorers(
    GATEWAY_ROUTE, [("coherent_synthesis", "orchestrator-workers-agent-coherent_synthesis")]
)

# The alias provisioning points at the "live" version of each step's prompt — see
# packages/mlflow-server/scripts/provision_prompts.py, which registers this agent's three step
# prompts from packages/mlflow-server/prompts/orchestrator-workers-agent/*.txt.
_PROMPT_ALIAS = PRODUCTION_ALIAS

STEPS = ("orchestrate", "worker", "synthesize")


class Subtask(BaseModel):
    """One subtask the orchestrator decided this task needs.

    Mirrors orchestrator-workers.md's example directly.
    """

    description: str = Field(description="What this subtask needs to accomplish.")


class TaskBreakdown(BaseModel):
    """Structured output for the orchestrator node.

    Not a fixed count of subtasks, but however many the specific input actually calls for.
    """

    analysis: str = Field(description="Why this decomposition fits the input.")
    subtasks: list[Subtask]


class OverallState(TypedDict):
    """State threaded through the graph.

    `subtasks` is written once by `orchestrate` and read by the routing function that fans out
    to workers — unlike `worker_results`, it isn't accumulated across nodes, so no reducer is
    needed. `worker_results` uses an `operator.add` reducer: every dynamically-spawned
    `run_worker` writes to this *same* key, so the reducer is what lets LangGraph accumulate all
    of their outputs — however many workers actually ran — instead of the last writer clobbering
    the rest (same convention as map_reduce_agent.graph.OverallState.jokes).
    """

    task: str
    analysis: str
    subtasks: list[str]
    worker_results: Annotated[list[str], operator.add]
    synthesis: str


class WorkerState(TypedDict):
    """Per-worker state: exactly what one `Send` call carries into `run_worker`.

    Deliberately narrower than `OverallState` — a worker only ever sees its own `subtask`, never
    the full breakdown or other workers' state, which is what keeps workers independent (see
    the docs page's case study on vague task descriptions causing duplicated/gapped work).
    """

    subtask: str


def load_step_prompt_version(step: str, *, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch one step's prompt version from the MLflow prompt registry.

    Thin wrapper around `agents_common.prompts.load_prompt_version`, binding this agent's
    per-step registry name (`<EXPERIMENT_NAME>-<step>`) and experiment — each step is registered
    as its own prompt name under this agent's single experiment; see provision_prompts.py's
    per-subdirectory provisioning.

    Returns the full `PromptVersion` (not just its text) so a caller can pass it to
    `link_prompts_to_trace` afterwards — see `orchestrator_workers_agent.__main__` for the
    intended usage.

    Args:
        step: One of "orchestrate", "worker", "synthesize".
        alias: Prompt registry alias to load. Defaults to the production alias.
    """
    return load_prompt_version(
        f"{EXPERIMENT_NAME}-{step}", experiment_name=EXPERIMENT_NAME, alias=alias
    )


def load_step_prompt(step: str, *, alias: str = _PROMPT_ALIAS) -> str:
    """Fetch one step's prompt text from the MLflow prompt registry.

    Thin wrapper around `load_step_prompt_version` for callers that only need the text (e.g.
    `build_orchestrator_graph`'s default path) and don't need to link the version to a trace
    afterwards.
    """
    return prompt_text(load_step_prompt_version(step, alias=alias))


def link_prompts_to_trace(prompt_versions: dict[str, PromptVersion], trace_id: str | None) -> None:
    """Link this invocation's step prompt versions to a trace.

    Thin wrapper around `agents_common.prompts.link_prompts_to_trace` that accepts this agent's
    step-keyed dict shape — see that function's docstring for `trace_id` semantics.
    """
    _link_prompts_to_trace(list(prompt_versions.values()), trace_id)


def build_orchestrator_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    step_prompts: dict[str, str] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the orchestrator-workers workflow.

    The caller owns the checkpointer's lifecycle, same convention as
    map_reduce_agent.graph.build_map_reduce_graph.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration)
            or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route every stage calls. Defaults to this package's own
            `GATEWAY_ROUTE`; overridable for tests against a different route.
        step_prompts: Overrides the registry-fetched step prompts, keyed by step name
            ("orchestrate", "worker", "synthesize"). Defaults to `None`, which fetches each
            step's current `production`-aliased prompt via `load_step_prompt()`. Pass literal
            strings in tests that need a hermetic build with no MLflow prompt-registry
            dependency.

    Returns:
        A compiled LangGraph graph, invoked with `{"task": ..., "analysis": "", "subtasks": [],
        "worker_results": [], "synthesis": ""}`. The number of `run_worker` calls spawned equals
        however many subtasks the orchestrator decides on for this specific `task` — not fixed
        by the graph's structure.
    """
    model = get_chat_model(gateway_route)
    orchestrator = model.with_structured_output(TaskBreakdown)
    prompts = (
        step_prompts
        if step_prompts is not None
        else {step: load_step_prompt(step) for step in STEPS}
    )

    def orchestrate(state: OverallState) -> dict[str, Any]:
        breakdown = orchestrator.invoke(f"{prompts['orchestrate']}\n\nTask: {state['task']}")
        assert isinstance(breakdown, TaskBreakdown)
        _logger.info("task_decomposed", subtask_count=len(breakdown.subtasks))
        return {
            "analysis": breakdown.analysis,
            "subtasks": [subtask.description for subtask in breakdown.subtasks],
        }

    def continue_to_workers(state: OverallState) -> list[Send]:
        # Runtime-determined fan-out, twice over: not just how many Sends (like map-reduce), but
        # what each one's payload actually says to do — both come from the orchestrator's own
        # structured output, not from a list the caller passed in.
        return [Send("run_worker", {"subtask": subtask}) for subtask in state["subtasks"]]

    def run_worker(state: WorkerState) -> dict[str, list[str]]:
        response = model.invoke(f"{prompts['worker']}\n\n{state['subtask']}")
        _logger.info("worker_completed", subtask=state["subtask"])
        return {"worker_results": [str(response.content)]}

    def synthesize(state: OverallState) -> dict[str, str]:
        combined = "\n\n".join(
            f"Subtask result {i}:\n{result}"
            for i, result in enumerate(state["worker_results"], start=1)
        )
        response = model.invoke(
            f"{prompts['synthesize']}\n\nOriginal task: {state['task']}\n\n{combined}"
        )
        _logger.info("synthesized", worker_result_count=len(state["worker_results"]))
        return {"synthesis": str(response.content)}

    graph = StateGraph(OverallState)
    graph.add_node("orchestrate", orchestrate)
    graph.add_node("run_worker", run_worker)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "orchestrate")
    graph.add_conditional_edges("orchestrate", continue_to_workers, ["run_worker"])
    graph.add_edge("run_worker", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str) -> dict[str, Any]:
    """Build the `.invoke()` config for a thread.

    Every call site that runs the compiled graph should route through this, same convention as
    react_agent.graph.invoke_config / map_reduce_agent.graph.invoke_config.
    """
    return {"configurable": {"thread_id": thread_id}}
