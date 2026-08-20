"""Builds the evaluator-optimizer workflow graph.

This is a workflow, not an agent, per Anthropic's framing (see
docs/patterns/agent/evaluator-optimizer.md): the shape — generate, evaluate, loop back or exit —
is fixed in code ahead of time, even though the *number of loop iterations* actually taken
varies with how many revisions the evaluator demands. Built with a raw LangGraph `StateGraph`
rather than `langchain.agents.create_agent` (contrast with `react_agent.graph`), since there's no
tool loop here for `create_agent` to compile — the loop here is generate/evaluate, not
model-driven tool calls.

Distinct from Self-Refine (not implemented in this repo): this pattern uses two separate LLM
calls in different roles — a generator and an evaluator — rather than one model critiquing its
own output. The evaluator's structured output (`Evaluation`) is what actually terminates the
loop: `approved=True`, or `iteration >= max_iterations` (never unbounded) — mirrors
evaluator-optimizer.md's own `evaluator_optimizer_loop` example directly, just expressed as a
LangGraph conditional edge instead of a Python `for` loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

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
    "DEFAULT_MAX_ITERATIONS",
    "GATEWAY_ROUTE",
    "PRODUCTION_SCORERS",
    "STEPS",
    "Evaluation",
    "OptimizerState",
    "build_evaluator_optimizer_graph",
    "invoke_config",
    "link_prompts_to_trace",
    "load_step_prompt",
    "load_step_prompt_version",
    "prompt_text",
]

# This agent's own MLflow experiment — see agents_common.observability.configure_mlflow.
EXPERIMENT_NAME = "evaluator-optimizer-agent"

# The MLflow AI Gateway route this agent calls — same route react-agent/orchestrator-workers-agent
# use, since this is a reference example rather than a production workload that needs its own
# provisioned model.
GATEWAY_ROUTE = "gpt-oss-120b"

# Scorers run continuously against a sampled slice of live production traces — see
# agents_common.observability.register_production_monitors, provisioned via
# packages/mlflow-server/scripts/provision_monitors.py. `meets_criteria`'s guideline text is
# loaded from packages/mlflow-server/judges/evaluator-optimizer-agent-meets_criteria.txt, the
# same source that eval suite loads it from — single source of truth, see agents_common.judges.
PRODUCTION_SCORERS: list[tuple[Any, float]] = build_production_scorers(
    GATEWAY_ROUTE, [("meets_criteria", "evaluator-optimizer-agent-meets_criteria")]
)

# The alias provisioning points at the "live" version of each step's prompt — see
# packages/mlflow-server/scripts/provision_prompts.py, which registers this agent's two step
# prompts from packages/mlflow-server/prompts/evaluator-optimizer-agent/*.txt.
_PROMPT_ALIAS = PRODUCTION_ALIAS

STEPS = ("generate", "evaluate")

# Caps the generate/evaluate loop so a stubborn evaluator (or a task that can never satisfy its
# own criteria) can't run forever — mirrors evaluator-optimizer.md's own
# `evaluator_optimizer_loop(..., max_iterations: int = 3)` default directly.
DEFAULT_MAX_ITERATIONS = 3


class Evaluation(BaseModel):
    """Structured output for the evaluate node.

    Mirrors evaluator-optimizer.md's example directly.
    """

    approved: bool = Field(
        description="True if the response meets every criterion, false otherwise."
    )
    feedback: str = Field(description="Specific, actionable feedback for improving the response.")


class OptimizerState(TypedDict):
    """State threaded through the graph.

    `iteration` counts completed `generate` calls, checked against `max_iterations` (closed over
    by the graph, not stored in state) by the conditional edge after `evaluate`. `feedback` is
    empty on the first pass through `generate` and holds the evaluator's critique on every pass
    after that — `generate` branches on whether it's empty to decide whether it's writing a first
    draft or revising against feedback.
    """

    task: str
    criteria: str
    response: str
    feedback: str
    approved: bool
    iteration: int


# One `PromptLoaders` bundle per step, each bound to this agent's per-step registry name
# (`<EXPERIMENT_NAME>-<step>`) and the production alias — see provision_prompts.py's
# per-subdirectory provisioning.
_step_loaders = {
    step: make_prompt_loaders(
        f"{EXPERIMENT_NAME}-{step}", experiment_name=EXPERIMENT_NAME, alias=_PROMPT_ALIAS
    )
    for step in STEPS
}


def load_step_prompt_version(step: str, *, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch one step's prompt version from the MLflow prompt registry.

    Thin wrapper around `agents_common.make_prompt_loaders`, binding this agent's per-step
    registry name (`<EXPERIMENT_NAME>-<step>`) and experiment — each step is registered as its
    own prompt name under this agent's single experiment; see provision_prompts.py's
    per-subdirectory provisioning.

    Returns the full `PromptVersion` (not just its text) so a caller can pass it to
    `link_prompts_to_trace` afterwards — see `evaluator_optimizer_agent.__main__` for the
    intended usage.

    Args:
        step: One of "generate", "evaluate".
        alias: Prompt registry alias to load. Defaults to the production alias.
    """
    if alias == _PROMPT_ALIAS:
        return _step_loaders[step].load_version()
    return make_prompt_loaders(
        f"{EXPERIMENT_NAME}-{step}", experiment_name=EXPERIMENT_NAME, alias=alias
    ).load_version()


def load_step_prompt(step: str, *, alias: str = _PROMPT_ALIAS) -> str:
    """Fetch one step's prompt text from the MLflow prompt registry.

    Thin wrapper around `load_step_prompt_version` for callers that only need the text (e.g.
    `build_evaluator_optimizer_graph`'s default path) and don't need to link the version to a
    trace afterwards.
    """
    return prompt_text(load_step_prompt_version(step, alias=alias))


def link_prompts_to_trace(prompt_versions: dict[str, PromptVersion], trace_id: str | None) -> None:
    """Link this invocation's step prompt versions to a trace.

    Thin wrapper around `agents_common.make_prompt_loaders`'s per-step `link_to_trace` that
    accepts this agent's step-keyed dict shape — see that function's docstring for `trace_id`
    semantics.
    """
    for step, prompt_version in prompt_versions.items():
        _step_loaders[step].link_to_trace(prompt_version, trace_id)


def build_evaluator_optimizer_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    step_prompts: dict[str, str] | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the evaluator-optimizer workflow.

    The caller owns the checkpointer's lifecycle, same convention as
    orchestrator_workers_agent.graph.build_orchestrator_graph.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration)
            or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route both generate and evaluate call. Defaults to this
            package's own `GATEWAY_ROUTE`; overridable for tests against a different route.
        step_prompts: Overrides the registry-fetched step prompts, keyed by step name
            ("generate", "evaluate"). Defaults to `None`, which fetches each step's current
            `production`-aliased prompt via `load_step_prompt()`. Pass literal strings in tests
            that need a hermetic build with no MLflow prompt-registry dependency.
        max_iterations: Upper bound on `generate` calls before the loop exits regardless of
            `approved`. Defaults to `DEFAULT_MAX_ITERATIONS`.

    Returns:
        A compiled LangGraph graph, invoked with `{"task": ..., "criteria": ..., "response": "",
        "feedback": "", "approved": False, "iteration": 0}`. The number of generate/evaluate
        cycles actually run varies per invocation — as soon as `evaluate` approves, or
        `max_iterations` is hit, the loop exits.
    """
    model = get_chat_model(gateway_route)
    evaluator = model.with_structured_output(Evaluation)
    prompts = (
        step_prompts
        if step_prompts is not None
        else {step: load_step_prompt(step) for step in STEPS}
    )

    def generate(state: OptimizerState) -> dict[str, Any]:
        if state["feedback"]:
            prompt = (
                f"{prompts['generate']}\n\nTask: {state['task']}\n\n"
                f"Previous feedback to address:\n{state['feedback']}"
            )
        else:
            prompt = f"{prompts['generate']}\n\nTask: {state['task']}"
        response = model.invoke(prompt)
        _logger.info("generated", iteration=state["iteration"] + 1)
        return {"response": str(response.content), "iteration": state["iteration"] + 1}

    def evaluate(state: OptimizerState) -> dict[str, Any]:
        prompt = (
            f"{prompts['evaluate']}\n\nTask: {state['task']}\n\nCriteria: {state['criteria']}"
            f"\n\nResponse:\n{state['response']}"
        )
        result = evaluator.invoke(prompt)
        assert isinstance(result, Evaluation)
        _logger.info("evaluated", approved=result.approved, iteration=state["iteration"])
        return {"approved": result.approved, "feedback": result.feedback}

    def route_after_evaluate(state: OptimizerState) -> str:
        # Terminates the loop on either condition — approval, or the iteration cap — never both
        # unbounded, matching evaluator-optimizer.md's `for _ in range(max_iterations)` loop.
        if state["approved"] or state["iteration"] >= max_iterations:
            _logger.info(
                "loop_exited",
                approved=state["approved"],
                iteration=state["iteration"],
                hit_max_iterations=state["iteration"] >= max_iterations,
            )
            return END
        return "generate"

    graph = StateGraph(OptimizerState)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", route_after_evaluate, ["generate", END])

    return graph.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str) -> dict[str, Any]:
    """Build the `.invoke()` config for a thread.

    Every call site that runs the compiled graph should route through this, same convention as
    react_agent.graph.invoke_config / orchestrator_workers_agent.graph.invoke_config.
    """
    return {"configurable": {"thread_id": thread_id}}
