# ADR 0001: Tech stack and framework-tiering for example-agents

- **Status:** Accepted
- **Date:** 2026-08-15
- **Owner:** Joseph Searle

## Context

This repository exists to show off, with working code, the agent patterns we actually use
(the ones sketched but not yet written up in the playbook under
`pages/development/ai/agents/*` — patterns, stack, standards, swarm, security). It needs a
tech stack decision that is opinionated enough to be a real reference, not just a pile of
options.

Two prior spikes in the playbook are directly relevant and this ADR treats them as evidence,
not as a mandate to copy verbatim:

- `spikes/langgraph-swarm-pattern-itz-15103` — compares the **supervisor** pattern
  (`core-support-agent`) against the **swarm** pattern (`core-swarm-agent`), and a plain
  **single ReAct agent** (`asktz-request-agent`). Conclusion: swarm suits workflows that are
  not strictly linear; supervisor suits workflows with a clear top-down chain of command;
  a single ReAct agent is enough when there's no multi-domain handoff at all.
- `spikes/deepagent-itz-19739` — a persistent, webhook-driven PR-review agent that needs
  long-horizon planning, iteration, and a filesystem. This is the shape of problem deepagents
  is built for, not a general-purpose agent framework.

The open question from the brief: standardise on `deepagents` for every pattern (simplicity),
or pick between LangChain / LangGraph / deepagents per pattern based on Anthropic's agent
design principles ("start with the simplest thing that could work; add complexity —
multi-step orchestration, multi-agent, persistent planning — only when a single well-prompted
call with tools demonstrably can't do the job")?

## Decision

**Tiered by pattern complexity, not a single framework for everything.** All three tiers sit
on the same LangGraph runtime underneath (`create_agent` is itself a compiled LangGraph graph),
so there's no integration tax to moving between tiers — checkpointing, streaming, and tracing
work identically at every level.

| Tier | Framework | Use for | Example in this repo |
|---|---|---|---|
| 1 | `langchain.agents.create_agent` (LangChain v1) | A single agent with tools in a ReAct loop. The default. Reach for this first, every time, and only leave it when you hit something it can't express. | `agents/patterns/react-agent` (implemented) |
| 2 | Raw LangGraph (`StateGraph`, `langgraph-supervisor`, `langgraph-swarm`) | Multiple cooperating agents: a supervisor routing to specialist subgraphs, or peer agents handing off via `Command(goto=...)`. Needed when one agent's tool-loop can't express the control flow, or you need custom state, HITL interrupts, or explicit inter-agent contracts. | `agents/patterns/supervisor-agent`, `agents/patterns/swarm-agent` (stubs, follow-up PRs) |
| 3 | `deepagents` | Long-horizon, planning-heavy work with a virtual filesystem and spawned subagents — the ITZ-19739 shape of problem (multi-turn PR review, iterating on generated tests, proactive follow-up over days not minutes). Not used for anything that fits in tiers 1–2; its planning/filesystem/subagent machinery is overhead you don't want to pay for a simple tool-calling agent. | `agents/examples/experiment-analysis-agent` (implemented) — MLflow AI Issue Discovery over another agent's traces: search in batches, refine hypotheses, write a report. Lives under `agents/examples/` rather than `agents/patterns/` since it doesn't compose into a standalone reusable pattern with its own doc, and it analyzes the other agents' traces rather than fitting the same-shaped-request loop `agents/patterns/*` demonstrates. |

**Rule of thumb when adding a new example agent:** start writing it as `create_agent` (tier 1).
Only move to tier 2 if you can point at a concrete requirement — multiple domains with
different tools/prompts, a workflow that can't be one ReAct loop, or a need for supervisor/swarm
routing. Only reach for deepagents if the task is long-horizon and planning-first, not just
"has several steps."

### MCP server connections: MLflow's MCP Registry, not hardcoded launch commands

`experiment-analysis-agent` was the repo's first agent to consume an MCP server at runtime
(`mlflow-mcp`, via `langchain-mcp-adapters`). It initially hardcoded that server's stdio launch
command, duplicating `.mcp.json`'s declaration of the same server for interactive Claude Code
use. Fixing that duplication went through two passes:

1. **First pass**: extract the shared stdio command/args into one Python constant both
   `.mcp.json` and the agent could point at. Worked, but still a hardcoded launch command
   somewhere.
2. **Second pass** (current): resolve every MCP server connection dynamically via
   [MLflow's MCP Registry](https://mlflow.org/docs/latest/genai/mcp-registry/) —
   `mlflow.genai.search_mcp_access_endpoints` at runtime, `.mcp.json` and agent code both
   pointing at the same registered endpoint instead of either hardcoding anything.

This forced a real infrastructure change, confirmed against the installed `mlflow==3.15.1` API
(not just its docs, which were incomplete on these points):

- `mlflow.genai.create_mcp_access_endpoint`'s `transport_type` only accepts
  `"streamable-http"`/`"sse"` — never `"stdio"`.
- `mlflow mcp run` (the CLI `.mcp.json` used) only speaks stdio, no transport flag.
- `mcp-server-milvus`'s own `--streamable-http` CLI mode hard-codes `host="localhost"` with no
  override — unreachable from any other container.
- The registry's REST API is pure catalog metadata; it doesn't proxy or host live MCP traffic.
  Registering a server doesn't make it reachable — it only catalogs a pointer to wherever it
  already is reachable.

So both `mlflow-mcp` and `milvus-mcp` needed converting into persistent streamable-http services
first (`packages/mlflow-server/mlflow_mcp_server.py`, `packages/milvus/milvus_mcp_server.py` —
each bypasses its respective CLI, calling the underlying `FastMCP` instance directly with
`host="0.0.0.0"`), each its own docker-compose service, before `packages/mlflow-server/scripts/
provision_mcp_registry.py` had anything connectable to register. `docs-langchain`/
`reference-langchain` needed no such conversion — already remote, third-party-hosted HTTP
servers; registering them is pure cataloging.

All four of `.mcp.json`'s servers are registered this way, not just the one `experiment-
analysis-agent` currently consumes — so the next agent that needs one (e.g. the planned
RAG-pattern agent using Milvus) resolves a connection instead of reintroducing a hardcoded
launch command on day one. See `agents_common.mcp_servers` for the resolver and
`provision_mcp_registry.py`'s docstring for the registered server list and the
host/container-reachability known-gap this setup still has.

## Language, packaging, monorepo layout

- **Python 3.12**, managed with **uv**. One **uv workspace** at the repo root
  (`[tool.uv.workspace] members = ["agents/*", "packages/*"]`) — one lockfile, one shared
  venv, but each agent and shared package is independently installable/buildable/dockerizable.
  This mirrors the monorepo-benefits analysis in the swarm-pattern spike (single source of
  truth for shared concerns, one Ruff/mypy config, one dependency-audit surface) without
  forcing every agent to deploy together — each `agents/*` package still gets its own
  `Dockerfile` and can be built/shipped independently.
- **`packages/agents-common`** — shared, non-agent-specific code every agent imports: MLflow
  tracing setup, the Postgres checkpointer/store factory, and typed settings (`pydantic-settings`)
  loaded from environment variables. This is where the "don't duplicate Redis/MCP/Langfuse
  setup across projects" lesson from the swarm spike gets applied up front.
- **`packages/mlflow-server`** — the self-hosted MLflow tracking server as its own Dockerfile,
  not bundled into an agent image.

## Observability: MLflow

- **Self-hosted MLflow** (`packages/mlflow-server`), backed by **Postgres** (not SQLite —
  SQLite is the zero-config default but isn't safe for concurrent writers) and a local
  artifact volume in dev (swap for S3/GCS/Azure Blob via `MLFLOW_ARTIFACT_ROOT` in a real
  deployment).
- **Tracing:** `mlflow.langchain.autolog()` — MLflow 3's LangChain/LangGraph autologging
  captures every LLM call, tool call, and graph step as a trace with no manual instrumentation
  in agent code; wired once in `packages/agents-common/src/agents_common/observability`.
- **Experiment tracking, one experiment per agent, not per repo:** `configure_mlflow()` takes
  an `experiment_name` argument rather than reading a shared name from `.env` — each agent
  defines its own as a constant in its own package (e.g. `react_agent.EXPERIMENT_NAME =
  "react-agent"`). `react-agent`, `supervisor-agent`, etc. are meaningfully different systems;
  one shared `example-agents` experiment would mix their runs and metrics into one
  undifferentiated stream, which defeats the point of trending eval quality per pattern.
- **Evals:** `mlflow.genai.evaluate()` is the eval-suite harness (see `agents/patterns/react-agent/tests/evals`
  for the test code itself) — LLM-as-judge and deterministic scorers, run against a fixed
  dataset, logged as a run in that agent's own experiment so eval history is queryable over time,
  not just pass/fail in CI logs. The dataset itself lives in MLflow's dataset registry at
  runtime, not in the agent's package — seeded from git-tracked JSONL at
  `packages/mlflow-server/datasets/<agent-name>.jsonl` and synced in via `make provision-datasets`
  (`packages/mlflow-server/scripts/provision_datasets.py`), the same git-source-synced-into-mlflow
  pattern `provision_gateway_route.py` already uses for the AI Gateway route. This makes the
  dataset visible/browsable in the MLflow UI for every dev, not just readable by the one test
  file that happens to load it.
  Known gap: `.github/workflows/eval.yml` runs against a real, pre-existing external MLflow
  instance (`secrets.MLFLOW_TRACKING_URI`), not the local docker-compose `mlflow` container, so
  neither this dataset-provisioning script nor `provision_gateway_route.py` runs in CI — both the
  gateway route and any eval dataset must be provisioned against that external instance
  separately. Not solved by this change.

## Model access: MLflow AI Gateway, not per-provider API keys

- Every agent gets its chat model through `agents_common.models.get_chat_model(gateway_route)`,
  which calls a named route on the self-hosted **MLflow AI Gateway** instead of constructing a
  provider client (`ChatAnthropic`, etc.) with a provider API key baked into this repo's config.
- The gateway is mounted directly on the MLflow tracking server itself (`mlflow[genai]>=3.1`,
  routes under `/gateway/mlflow/v1`) and speaks the OpenAI Chat Completions format, so the
  client is a plain `ChatOpenAI` pointed at that base URL (`Settings.mlflow_gateway_base_url`)
  — not a MLflow-specific LangChain integration. `ChatMlflow`
  (`langchain_community.chat_models`), which this repo used before, talks to the *old*
  standalone MLflow 2.x gateway server (`mlflow gateway start`), which no longer exists in
  MLflow 3.x — the OpenAI-compatible route above is its replacement.
- **`MLFLOW_TRACKING_TOKEN`** authenticates tracking, `mlflow.genai.evaluate()`, and calls *to*
  the gateway alike — one bearer token, only required if the tracking server has auth enabled.
  It is distinct from `SELFHOSTED_MODEL_API_KEY`, which the *gateway itself* uses for its
  outbound call to the self-hosted model — see below.
- Which model actually answers a given route (a self-hosted vLLM/TGI deployment, etc.) is a
  gateway-side config concern, decided when the route is provisioned — agent code only ever
  names a route (`GATEWAY_ROUTE` in `react_agent/graph.py`), the same pattern used for
  `EXPERIMENT_NAME` above. Swapping the underlying model doesn't touch agent code.
- Routes aren't provisioned automatically. The gateway exposes a REST API for it (secret →
  model definition → endpoint → attach), normally driven through the MLflow web UI;
  `packages/mlflow-server/scripts/provision_gateway_route.py` (`make provision-gateway`) does
  the same four calls from the CLI so local setup is scriptable — see `.env.example`'s
  `SELFHOSTED_MODEL_*` / `GATEWAY_ROUTE_NAME` vars.
- If your gateway speaks a different client contract than `ChatOpenAI` expects,
  `agents_common/models/__init__.py` is the single place that changes — every agent goes
  through this one factory function, by design.

## Vector store: Milvus, standalone + Attu

- **`packages/milvus`** — self-hosted Milvus standalone (etcd + MinIO + the Milvus server
  itself), wired into `docker-compose.yml` the same way `packages/mlflow-server` is, mirroring
  the reference local setup at `~/milvus-standalone` (same images, same startup flags). Unlike
  `mlflow-server`, it needs no custom-built image — the official `milvusdb/milvus` image is a
  complete standalone server on its own — so this package is compose wiring and documentation,
  with a place reserved for a future `milvus.yaml`/analyzer config file if one is ever needed.
- **Attu** (`zilliz/attu`, pinned to `v2.5.6`) as the web UI for browsing collections and
  running ad hoc queries against Milvus without writing PyMilvus code — the same "give the dev a
  UI, not just an API" reasoning as running MLflow's own UI rather than only its tracking API.
- **Named Docker volumes**, not the bind-mounted `./volumes/` the reference setup uses — matches
  how `postgres-data` and `mlflow-artifacts` are already declared, and keeps a live vector index
  from ending up half-committed to git the way a bind-mounted folder can.
- **`Settings.milvus_uri`** joins `postgres_uri` and `mlflow_tracking_uri` in
  `agents_common.config` even though no example agent talks to Milvus yet — a future
  RAG-pattern agent (tier 1 or 2, depending on whether retrieval needs its own subgraph step)
  gets the same typed, one-place-to-change value instead of inventing a second convention later.
  `pymilvus`/`langchain-milvus` stay out of `agents-common`'s own dependencies until an agent
  actually needs them, same reasoning as `langchain-anthropic` never living there either.

## Checkpointing & memory: Postgres

- **`langgraph-checkpoint-postgres`** — `PostgresSaver` for short-term (per-thread) checkpointing
  and `PostgresStore` for long-term, cross-thread memory. One Postgres instance in
  `docker-compose.yml`, two logical databases (`agents` for checkpoints/store,
  `mlflow` for the tracking server's backend store) so the two concerns don't share a schema.
  Postgres doesn't create a database on first connection the way SQLite creates a file — the
  official image only auto-creates the one database named by `POSTGRES_DB` (`agents`) on first
  boot, so `infra/postgres/init.sql` creates the second (`mlflow`) via Postgres's
  `docker-entrypoint-initdb.d` convention. MLflow itself still owns everything *inside* that
  database — its own tables (experiments, runs, params, metrics, ...) are created and migrated
  by MLflow's own startup migrations; `init.sql` only makes the empty database exist for it to
  connect to.
- Chosen over SQLite/in-memory because every pattern example needs to demonstrate durable,
  resumable execution — that's the whole point of showing the checkpointing pattern, not an
  incidental detail.
- **`Settings.postgres_uri` is a computed property**, built from five individual fields
  (`POSTGRES_USER`/`PASSWORD`/`HOST`/`PORT`/`DB`) rather than one connection-string field —
  rotating a password or moving to a managed instance means changing one value in `.env`
  instead of also hand-editing a separate pre-assembled DSN that can drift out of sync with it.
- **pgAdmin** (`dpage/pgadmin4`, pinned to `9.17`) as the web UI for browsing checkpoints,
  memory, and MLflow's own tables — the same "give the dev a UI, not just a connection string"
  reasoning as running Attu alongside Milvus. `infra/postgres/servers.json` pre-registers this
  compose's `postgres` service on first boot (host, port, database, username), so the only
  manual step is entering the Postgres password once. pgAdmin's own login is a hardcoded dev
  credential in `docker-compose.yml`, not read from `.env` — it authenticates the UI itself,
  not Postgres, so it isn't a secret worth plumbing through config (same treatment as MinIO's
  `minioadmin`/`minioadmin` behind Milvus).

## SDLC tooling

Following the org's Python standards (`agentcraft:python-standards`):

| Concern | Tool | Notes |
|---|---|---|
| Packaging / envs | **uv** | workspace mode, `uv sync --frozen` in CI and Docker builds |
| Lint + format | **Ruff** | one root config, pre-commit + CI |
| Type checking | **mypy `--strict`** in CI, **pyright** in editor | mypy not in pre-commit (needs full project graph + real deps) |
| Tests | **pytest** + `pytest-asyncio` + **Hypothesis** | markers: `unit`, `integration`, `eval` |
| Docstrings | Google convention | enforced via Ruff `D` rules |
| Pre-commit | Ruff + file hygiene only | mypy/pytest stay in CI for speed |

Coverage floor is set to 80% (`docs/decisions` template default is 90%, but this is a
reference-examples repo, not a published library — deliberately relaxed and called out here
rather than silently diverging).

## CI/CD: GitHub Actions

Three workflows, deliberately not one, because they have very different cost/speed/blocking
profiles:

1. **`ci.yml`** — every push/PR. `uv sync --frozen`, Ruff lint+format check, mypy, unit tests
   (`-m unit`). No external services, no LLM calls, no tokens spent. This is the required
   status check for merges.
2. **`integration.yml`** — every push/PR. Spins up a `postgres` service container, runs
   `-m integration` (checkpointer round-trips, MLflow client against a real tracking server
   started in the job). Still no paid LLM calls — integration tests mock the model boundary.
3. **`eval.yml`** — MLflow GenAI eval suite (`-m eval`). Calls a real model and costs tokens,
   so it does **not** block every PR: runs on a nightly schedule, on-demand via
   `workflow_dispatch`, and on PRs labelled `run-evals`. Results are logged to the MLflow
   tracking server so eval quality is trended over time, and the job fails if scores regress
   past a threshold — this is the quality gate before a pattern gets called "reference-quality."

## Containerization

`docker-compose.yml` at the repo root runs the whole stack for local development:
`postgres` → `mlflow` (`packages/mlflow-server`) → agent services (`agents/patterns/react-agent`, ...).
Each agent has its own multi-stage `Dockerfile` built with `uv sync --frozen --no-dev`, so an
agent image only ever contains its own dependencies, not the whole workspace's.

## Consequences

- Adding a new pattern example means picking a tier from the table above and justifying it in
  that PR's description — this is meant to keep tier 2/3 from becoming the default out of habit.
- The eval workflow being non-blocking means a regression can land and only get caught
  overnight or when someone remembers the `run-evals` label — acceptable trade-off for token
  cost today; revisit if/when eval calls get cheap enough to run on every PR.
- Two logical Postgres databases in one instance is a dev-environment simplification; a real
  deployment would likely split MLflow's backend store onto managed Postgres separately from
  agent checkpoint storage.
