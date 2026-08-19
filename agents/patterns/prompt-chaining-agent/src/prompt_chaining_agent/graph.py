"""Builds the prompt chaining workflow graph.

This is a workflow, not an agent, per Anthropic's framing (see
docs/patterns/agent/prompt-chaining.md): the sequence of steps — outline, gate check, draft,
polish — is fixed in code ahead of time. Built with a raw LangGraph `StateGraph` rather than
`langchain.agents.create_agent` (contrast with `react_agent.graph`), since there's no tool loop
or LLM-decided branching here for `create_agent` to compile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from agents_common import get_chat_model
from agents_common.judges import build_production_scorers
from agents_common.prompts import (
    PRODUCTION_ALIAS,
    link_prompts_to_trace as _link_prompts_to_trace,
    load_prompt_version,
    prompt_text,
)
from langgraph.graph import END, START, StateGraph
import structlog

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from mlflow.entities.model_registry import PromptVersion

_logger = structlog.get_logger(__name__)

__all__ = [
    "PRODUCTION_SCORERS",
    "STEPS",
    "ChainState",
    "build_chain",
    "gate_check_outline",
    "invoke_config",
    "link_prompts_to_trace",
    "load_step_prompt",
    "load_step_prompt_version",
    "prompt_text",
]

# This agent's own MLflow experiment — see agents_common.observability.configure_mlflow.
EXPERIMENT_NAME = "prompt-chaining-agent"

# The MLflow AI Gateway route this agent calls — same route react-agent uses, since this is a
# reference example rather than a production workload that needs its own provisioned model.
GATEWAY_ROUTE = "gpt-oss-120b"

_JUDGE_MODEL_URI = f"openai:/{GATEWAY_ROUTE}"

# Scorers run continuously against a sampled slice of live production traces — see
# agents_common.observability.register_production_monitors, provisioned via
# packages/mlflow-server/scripts/provision_monitors.py. `build_production_scorers` derives the
# `gateway:/<route>` judge model `Scorer.start()` requires — see react_agent.graph's
# PRODUCTION_SCORERS comment for why this differs from `_JUDGE_MODEL_URI`'s `openai:/<route>`
# form. `well_formed_prose`'s guideline text is loaded from
# packages/mlflow-server/judges/prompt-chaining-agent-well_formed_prose.txt, the same source that
# eval suite loads it from — single source of truth, see agents_common.judges.
PRODUCTION_SCORERS: list[tuple[Any, float]] = build_production_scorers(
    GATEWAY_ROUTE, [("well_formed_prose", "prompt-chaining-agent-well_formed_prose")]
)

# The alias provisioning points at the "live" version of each step prompt — see
# packages/mlflow-server/scripts/provision_prompts.py, which registers this agent's three step
# prompts from packages/mlflow-server/prompts/prompt-chaining-agent/*.txt.
_PROMPT_ALIAS = PRODUCTION_ALIAS

STEPS = ("outline", "draft", "polish")

# Minimum number of non-empty outline lines before the chain is allowed to continue into
# drafting — see docs/patterns/agent/prompt-chaining.md's gate_check_outline example, which
# this mirrors directly.
_MIN_OUTLINE_SECTIONS = 3


class ChainState(TypedDict):
    """State threaded through the chain. Each node reads the previous step's field."""

    topic: str
    outline: str
    draft: str
    final: str


def load_step_prompt_version(step: str, *, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch one step's prompt version from the MLflow prompt registry.

    Thin wrapper around `agents_common.prompts.load_prompt_version`, binding this agent's
    per-step registry name (`<EXPERIMENT_NAME>-<step>`) and experiment — each step is registered
    as its own prompt name under this agent's single experiment; see provision_prompts.py's
    per-subdirectory provisioning.

    Returns the full `PromptVersion` (not just its text) so a caller can pass it to
    `link_prompts_to_trace` afterwards — see `prompt_chaining_agent.__main__` for the intended
    usage.

    Args:
        step: One of "outline", "draft", "polish".
        alias: Prompt registry alias to load. Defaults to the production alias.
    """
    return load_prompt_version(
        f"{EXPERIMENT_NAME}-{step}", experiment_name=EXPERIMENT_NAME, alias=alias
    )


def load_step_prompt(step: str, *, alias: str = _PROMPT_ALIAS) -> str:
    """Fetch one step's prompt text from the MLflow prompt registry.

    Thin wrapper around `load_step_prompt_version` for callers that only need the text (e.g.
    `build_chain`'s default path) and don't need to link the version to a trace afterwards.
    """
    return prompt_text(load_step_prompt_version(step, alias=alias))


def link_prompts_to_trace(prompt_versions: dict[str, PromptVersion], trace_id: str | None) -> None:
    """Link this invocation's step prompt versions to a trace.

    Thin wrapper around `agents_common.prompts.link_prompts_to_trace` that accepts this agent's
    step-keyed dict shape — see that function's docstring for `trace_id` semantics.
    """
    _link_prompts_to_trace(list(prompt_versions.values()), trace_id)


def gate_check_outline(outline: str) -> None:
    """Fail fast on a malformed or too-thin outline before it reaches the drafting step.

    Raises:
        ValueError: The outline has fewer than `_MIN_OUTLINE_SECTIONS` non-empty lines.
    """
    sections = [line for line in outline.splitlines() if line.strip()]
    if len(sections) < _MIN_OUTLINE_SECTIONS:
        msg = (
            f"Outline only has {len(sections)} section(s) — expected at least "
            f"{_MIN_OUTLINE_SECTIONS} before drafting."
        )
        _logger.warning("outline_gate_failed", section_count=len(sections))
        raise ValueError(msg)
    _logger.info("outline_gate_passed", section_count=len(sections))


def build_chain(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    gateway_route: str = GATEWAY_ROUTE,
    step_prompts: dict[str, str] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct and compile the prompt chaining workflow.

    The caller owns the checkpointer's lifecycle, same convention as react_agent.graph.build_agent.

    Args:
        checkpointer: A LangGraph checkpointer, e.g. a `PostgresSaver` (production/integration)
            or `InMemorySaver` (unit tests).
        gateway_route: MLflow AI Gateway route to call for this chain's model. Defaults to
            this package's own `GATEWAY_ROUTE`; overridable for tests against a different route.
        step_prompts: Overrides the registry-fetched step prompts, keyed by step name ("outline",
            "draft", "polish"). Defaults to `None`, which fetches each step's current
            `production`-aliased prompt via `load_step_prompt()`. Pass literal strings in tests
            that need a hermetic build with no MLflow prompt-registry dependency.

    Returns:
        A compiled LangGraph graph, invoked with `{"topic": ..., "outline": "", "draft": "",
        "final": ""}`.
    """
    model = get_chat_model(gateway_route)
    prompts = (
        step_prompts
        if step_prompts is not None
        else {step: load_step_prompt(step) for step in STEPS}
    )

    def generate_outline(state: ChainState) -> dict[str, str]:
        response = model.invoke(f"{prompts['outline']}\n\n{state['topic']}")
        _logger.info("step_completed", step="outline")
        return {"outline": str(response.content)}

    def check_outline(state: ChainState) -> dict[str, str]:
        gate_check_outline(state["outline"])
        return {}

    def write_draft(state: ChainState) -> dict[str, str]:
        response = model.invoke(f"{prompts['draft']}\n\n{state['outline']}")
        _logger.info("step_completed", step="draft")
        return {"draft": str(response.content)}

    def polish_draft(state: ChainState) -> dict[str, str]:
        response = model.invoke(f"{prompts['polish']}\n\n{state['draft']}")
        _logger.info("step_completed", step="polish")
        return {"final": str(response.content)}

    graph = StateGraph(ChainState)
    graph.add_node("generate_outline", generate_outline)
    graph.add_node("check_outline", check_outline)
    graph.add_node("write_draft", write_draft)
    graph.add_node("polish_draft", polish_draft)

    graph.add_edge(START, "generate_outline")
    graph.add_edge("generate_outline", "check_outline")
    graph.add_edge("check_outline", "write_draft")
    graph.add_edge("write_draft", "polish_draft")
    graph.add_edge("polish_draft", END)

    return graph.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str) -> dict[str, Any]:
    """Build the `.invoke()` config for a thread.

    Every call site that runs the compiled graph should route through this, same convention as
    react_agent.graph.invoke_config (no recursion cap here — the chain's step count is fixed).
    """
    return {"configurable": {"thread_id": thread_id}}
