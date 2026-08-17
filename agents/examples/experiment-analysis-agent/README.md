# experiment-analysis-agent — Tier 3: deep agent

The repo's first `deepagents` pattern — see `docs/decisions/0001-tech-stack.md`'s tiering table.
Runs [MLflow AI Issue Discovery](https://mlflow.org/docs/latest/genai/eval-monitor/ai-insights/ai-issue-discovery/)
against another agent's MLflow experiment: search its traces in batches, form and refine
hypotheses about operational and quality issues across those batches, and write a markdown
report — long-horizon, planning-heavy analysis that doesn't fit a single ReAct loop (Tier 1) or
a fixed multi-agent control flow (Tier 2).

Two deliberate deviations from this repo's usual per-pattern checklist: no `docker-compose.yml`
service (this is a batch job, not a long-running service) and no `tests/evals` (its output is an
exploratory report, not a graded answer against a fixed dataset — see `tests/unit`/
`tests/integration` instead).

## What it demonstrates

- `deepagents.create_deep_agent` — this repo's first use of the framework, with the same
  self-hosted MLflow AI Gateway model every other pattern uses (`GATEWAY_ROUTE` in
  `src/experiment_analysis_agent/graph.py`, via `agents_common.get_chat_model()`), not a
  provider-specific model string.
- MCP tools wired into a deep agent: `langchain_mcp_adapters.MultiServerMCPClient` connects to
  the same local `mlflow-mcp` server `.mcp.json` already defines, and the tool list is filtered
  down to `MLFLOW_MCP_TOOL_ALLOWLIST` — six read-only tools (`search_traces`, `get_trace`, ...)
  — before being handed to `create_deep_agent`, so mutating tools never reach the agent.
  See `graph.py`'s `_mlflow_mcp_client`/`_filter_allowlisted_tools`.
- The deep agent's virtual filesystem as the report-writing surface: the system prompt tells the
  agent to `write_file` its findings to `REPORT_PATH`, and `__main__.py` reads that back out of
  the final state (`result["files"][REPORT_PATH]`) and flushes it to a real file on disk.
- Registry-backed prompt with template variables: the system prompt isn't a Python string —
  it's checked into `packages/mlflow-server/prompts/experiment-analysis-agent.txt`, provisioned
  into MLflow's prompt registry (`make provision-prompts`), and fetched + rendered at runtime
  (`load_system_prompt_version()` + `render_system_prompt()` in `src/experiment_analysis_agent/graph.py`).
  Unlike every other agent's static prompt, this one has `{{target_experiment}}`/`{{report_path}}`
  template variables (MLflow's own `{{var}}` templating, via `PromptVersion.format()`), since the
  target experiment is only known at invocation time.

## Run it

```bash
# From the repo root, with the workspace synced and Postgres + MLflow running (`make up`), and
# traces already present in the target experiment (e.g. `make demo`):
make analyze-experiment EXPERIMENT=react-agent   # writes report-react-agent.md

# Or directly:
uv run --package experiment-analysis-agent experiment-analysis-agent react-agent
```

## Test it

```bash
uv run pytest -m unit          # fast, no external services — tool-allowlist filtering, prompt assembly
uv run pytest -m integration   # needs `make up` + a real trace (e.g. `make demo` first)
```
