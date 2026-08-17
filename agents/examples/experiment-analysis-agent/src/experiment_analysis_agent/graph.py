"""Builds the experiment-analysis deep agent.

This is the repo's first Tier 3 (`deepagents`) pattern — see docs/decisions/0001-tech-stack.md.
Tiers 1-2 (a single ReAct loop, or a fixed multi-agent control flow) don't fit here: MLflow AI
Issue Discovery (https://mlflow.org/docs/latest/genai/eval-monitor/ai-insights/ai-issue-discovery/)
is long-horizon and planning-heavy — search a target experiment's traces in batches, form and
refine hypotheses about operational/quality issues across those batches, then write a structured
report — exactly the "virtual filesystem + planning" shape `deepagents` exists for.

Unlike every other pattern in this repo, this agent doesn't get its own `PRODUCTION_SCORERS`
entry in `provision_monitors.py` — it's tooling that analyzes the *other* agents' traces, not a
production agent whose own traces need continuous quality scoring.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agents_common import get_chat_model
from agents_common.config import Settings, get_settings
from agents_common.mcp_servers import mlflow_mcp_connection
from agents_common.prompts import (
    PRODUCTION_ALIAS,
    link_prompts_to_trace,
    load_prompt_version,
)
from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain.agents.middleware import wrap_model_call
from langchain_mcp_adapters.client import MultiServerMCPClient
import mlflow

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware import ModelRequest, ModelResponse
    from langchain_core.messages import AnyMessage
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph
    from mlflow.entities.model_registry import PromptVersion

__all__ = [
    "DEFAULT_RECURSION_LIMIT",
    "EXPERIMENT_NAME",
    "GATEWAY_ROUTE",
    "MLFLOW_MCP_TOOL_ALLOWLIST",
    "REPORT_PATH",
    "build_agent",
    "link_prompt_to_trace",
    "load_system_prompt_version",
    "render_system_prompt",
]

# `create_deep_agent` auto-adds a "general-purpose" subagent (exposed via the `task` tool)
# unless disabled — and a subagent's model calls do NOT inherit the parent's `middleware=` list
# (see `create_deep_agent`'s docs on `subagents`), so `_flatten_content_blocks_for_gateway`
# below never ran for it. Confirmed live: the model delegated a step via `task`, that subagent's
# own tool-error content came back as a content-blocks list, and the same gateway 400 this
# module's middleware exists to prevent recurred — just unprotected. This agent's analysis is
# sequential (search traces, batch by batch, against one MCP server) with no benefit from
# subagent parallelism, so disabling the tool entirely is simpler and more robust than trying to
# thread the workaround into a subagent spec too. Registered under the "openai" provider key
# (not a specific model) since this is the only deepagents user in this repo and always calls an
# OpenAI-compatible model via the gateway — see agents_common.models.get_chat_model.
register_harness_profile(
    "openai",
    HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
)

# This agent's own MLflow experiment — see agents_common.observability.configure_mlflow for why
# this is a per-agent constant. Meta-tooling still traces itself, same as every other pattern.
EXPERIMENT_NAME = "experiment-analysis-agent"

# Same self-hosted gateway route every other pattern in this repo calls — see
# agents_common.models.get_chat_model. No provider API key held directly here either.
GATEWAY_ROUTE = "gpt-oss-120b"

# Read-only mlflow-mcp tools this analysis actually needs. Filtering `MultiServerMCPClient`'s
# `get_tools()` down to this set (rather than handing the agent every tool the server exposes)
# keeps mutating tools like `delete_traces`/`delete_experiment` out of reach — the same intent
# the old Claude-CLI stand-in expressed via `--allowedTools`, now enforced in code instead of a
# shell-level permission flag.
MLFLOW_MCP_TOOL_ALLOWLIST = frozenset(
    {
        "search_experiments",
        "get_experiment",
        "search_traces",
        "get_trace",
        "get_trace_assessment",
        "list_scorers",
    }
)

# Where the deep agent writes its findings in its virtual filesystem — __main__.py reads this
# back out of the final state and flushes it to a real file on disk.
REPORT_PATH = "/report.md"

# A genuine multi-batch trace analysis (resolve the experiment, several rounds of search_traces
# + get_trace, list_scorers, then write_file) needs far more steps than react_agent's simple
# tool-loop — its DEFAULT_RECURSION_LIMIT of 25 isn't enough here and was observed truncating
# a real run before it ever reached the write_file step. LangGraph enforces this as a step count
# on the compiled graph, set per-invocation via `config`, not on `create_deep_agent` itself.
DEFAULT_RECURSION_LIMIT = 100

_PROMPT_ALIAS = PRODUCTION_ALIAS


def load_system_prompt_version(*, alias: str = _PROMPT_ALIAS) -> PromptVersion:
    """Fetch this agent's system prompt version from the MLflow prompt registry.

    Same convention as every other pattern in this repo: the prompt isn't a Python string, it's
    checked into `packages/mlflow-server/prompts/experiment-analysis-agent.txt`, provisioned via
    `make provision-prompts`, and fetched here — see `react_agent.graph.load_system_prompt_version`
    for the pattern this mirrors. Unlike the static prompts every other agent registers, this one
    has `{{target_experiment}}`/`{{report_path}}` template variables (MLflow's own `{{var}}`
    templating — see `PromptVersion.format`), since the target experiment is only known at
    invocation time — so callers use `render_system_prompt` instead of the shared
    `agents_common.prompts.prompt_text` helper, which returns the raw unformatted template.

    Returns the full `PromptVersion` (not just its text) so a caller can pass it to
    `link_prompt_to_trace` afterwards — see `experiment_analysis_agent.__main__`.
    """
    return load_prompt_version(EXPERIMENT_NAME, experiment_name=EXPERIMENT_NAME, alias=alias)


def render_system_prompt(prompt_version: PromptVersion, *, target_experiment: str) -> str:
    """Fill in this prompt's `{{target_experiment}}`/`{{report_path}}` template variables."""
    formatted = prompt_version.format(target_experiment=target_experiment, report_path=REPORT_PATH)
    if not isinstance(formatted, str):
        msg = f"Expected a plain-text prompt template for {prompt_version.name!r}, got {type(formatted).__name__}"
        raise TypeError(msg)
    return formatted


def link_prompt_to_trace(prompt_version: PromptVersion, trace_id: str | None) -> None:
    """Link this run's prompt version to its trace — see `agents_common.prompts.link_prompts_to_trace`."""
    link_prompts_to_trace([prompt_version], trace_id)


def _mlflow_mcp_client(settings: Settings) -> MultiServerMCPClient:
    """Connect to the `mlflow-mcp` service, resolved via MLflow's MCP Registry.

    `agents_common.mcp_servers.mlflow_mcp_connection` looks up the registered access endpoint
    (a persistent streamable-http service, not a stdio subprocess this function spawns itself)
    — see that module's docstring for why.
    """
    return MultiServerMCPClient({"mlflow-mcp": mlflow_mcp_connection(settings)})


def _filter_allowlisted_tools(tools: list[BaseTool]) -> list[BaseTool]:
    return [tool for tool in tools if tool.name in MLFLOW_MCP_TOOL_ALLOWLIST]


def _flatten_text_only_content(content: str | list[Any]) -> str | list[Any]:
    """Collapse a text-only content-blocks list back into a plain string.

    LangChain v1 always represents `ModelRequest.system_message`'s content as a list of
    content blocks internally — even a plain-string `system_prompt` gets wrapped as
    `[{"type": "text", "text": ...}]` (see "Standard content blocks" in LangChain's docs).
    That's a valid OpenAI Chat Completions request shape, but our self-hosted AI Gateway's
    backend implements a stricter schema that only accepts `content` as a bare string — it
    400s on a list, even a single-element one. Only collapses blocks that are *all* plain text
    (no images, tool-result blocks, etc.), leaving anything else untouched.
    """
    if isinstance(content, str):
        return content
    if content and all(
        isinstance(block, dict) and block.get("type") == "text" for block in content
    ):
        return "".join(block.get("text", "") for block in content)
    return content


def _flatten_message_content(message: AnyMessage) -> AnyMessage:
    flattened = _flatten_text_only_content(message.content)
    if flattened is message.content:
        return message
    return message.model_copy(update={"content": flattened})


def _flattened_request(request: ModelRequest) -> ModelRequest:
    new_messages = [_flatten_message_content(message) for message in request.messages]
    new_system_message = (
        None
        if request.system_message is None
        else request.system_message.model_copy(
            update={"content": _flatten_text_only_content(request.system_message.content)}
        )
    )
    return request.override(messages=new_messages, system_message=new_system_message)


@wrap_model_call
async def _flatten_content_blocks_for_gateway(
    request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
) -> ModelResponse:
    """Work around the self-hosted gateway rejecting list-form message content.

    See `_flatten_text_only_content`. Scoped to this one Tier 3 pattern rather than
    `agents_common.get_chat_model` globally: Tier 1/2 patterns in this repo never trigger it,
    since `create_agent` doesn't assemble multi-block system messages the way `deepagents` does.

    Async, not sync: `build_agent`/`run_analysis` always invoke via `ainvoke`, and
    `@wrap_model_call` on a sync function only provides a sync `wrap_model_call` hook — an
    `ainvoke()` run then has no `awrap_model_call` to call and raises `NotImplementedError`.
    """
    return await handler(_flattened_request(request))


async def build_agent(
    *,
    target_experiment: str,
    gateway_route: str = GATEWAY_ROUTE,
    system_prompt: str | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Construct the deep agent: gateway model + allowlisted mlflow-mcp tools + analysis prompt.

    Args:
        target_experiment: The MLflow experiment name to analyze, e.g. "react-agent".
        gateway_route: MLflow AI Gateway route to call for this agent's model. Defaults to this
            package's own `GATEWAY_ROUTE`; overridable for tests.
        system_prompt: Overrides the registry-fetched prompt. Defaults to `None`, which fetches
            the current `production`-aliased prompt via `load_system_prompt_version` and renders
            it with `target_experiment` — the normal runtime path. Pass a literal string in tests
            that need a hermetic build with no MLflow prompt-registry dependency; callers that
            need the fetched `PromptVersion` for `link_prompt_to_trace` afterwards (see
            `experiment_analysis_agent.__main__`) should fetch and render it themselves and pass
            the result here, same as `react_agent.__main__`'s pattern.
        settings: Override settings; defaults to `get_settings()`.
    """
    settings = settings or get_settings()
    client = _mlflow_mcp_client(settings)
    tools = _filter_allowlisted_tools(await client.get_tools())

    if system_prompt is None:
        system_prompt = render_system_prompt(
            load_system_prompt_version(), target_experiment=target_experiment
        )

    return create_deep_agent(
        model=get_chat_model(gateway_route, settings=settings),
        tools=tools,
        system_prompt=system_prompt,
        middleware=[_flatten_content_blocks_for_gateway],
        name="experiment-analysis-agent",
    )


def run_analysis(
    target_experiment: str,
    *,
    prompt_version: PromptVersion | None = None,
    settings: Settings | None = None,
) -> str:
    """Synchronous convenience wrapper: build, run once, return the report text.

    Reads the report back out of the final state's virtual filesystem (`DeepAgentState`'s
    `files` field) — this deep agent never touches the real filesystem, so the caller is
    responsible for flushing `REPORT_PATH` to disk if a persisted file is wanted.

    Args:
        target_experiment: The MLflow experiment name to analyze.
        prompt_version: Pass the `PromptVersion` from `load_system_prompt_version()` to both
            render this run's system prompt from it and link it to the resulting trace
            afterwards (see `experiment_analysis_agent.__main__`). Leave `None` (e.g. in tests)
            to let `build_agent` fetch its own default and skip trace-linking.
        settings: Override settings; defaults to `get_settings()`.
    """
    return asyncio.run(
        _run_analysis_async(target_experiment, prompt_version=prompt_version, settings=settings)
    )


async def _run_analysis_async(
    target_experiment: str, *, prompt_version: PromptVersion | None, settings: Settings | None
) -> str:
    system_prompt = (
        None
        if prompt_version is None
        else render_system_prompt(prompt_version, target_experiment=target_experiment)
    )
    agent = await build_agent(
        target_experiment=target_experiment, system_prompt=system_prompt, settings=settings
    )
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze experiment '{target_experiment}' and write the report.",
                }
            ]
        },
        config={
            "configurable": {"thread_id": target_experiment},
            "recursion_limit": DEFAULT_RECURSION_LIMIT,
        },
    )

    if prompt_version is not None:
        link_prompt_to_trace(prompt_version, mlflow.get_last_active_trace_id())

    # `files` is `dict[path, FileData]`, not `dict[path, str]` — FileData is a TypedDict with
    # `content`/`encoding`/`created_at`/`modified_at` (see deepagents' backends docs). Content is
    # always `"utf-8"`-encoded plain text for a report our own prompt writes via `write_file`.
    files = result.get("files", {})
    if REPORT_PATH not in files:
        msg = f"Agent finished without writing a report to {REPORT_PATH}"
        raise RuntimeError(msg)
    return str(files[REPORT_PATH]["content"])
