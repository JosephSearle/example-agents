<div align="center">

# Example Agents

Reference implementations of the agent patterns this team uses — real, running code, not diagrams.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-workspace-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![LangChain](https://img.shields.io/badge/LangChain-create__agent-1C3C3C?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-workflows-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![MLflow](https://img.shields.io/badge/MLflow-AI%20Gateway%20%2B%20Evals-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-checkpointing-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Milvus](https://img.shields.io/badge/Milvus-vector%20store-00A1EA?logo=milvus&logoColor=white)](https://milvus.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Ruff](https://img.shields.io/badge/lint%2Fformat-Ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/types-mypy%20strict-blue)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## Highlights

- **Companion to the playbook.** Backs the (currently stub) standards pages under
  `pages/development/ai/agents/*` with working reference code until those are filled in.
- **Framework tiering, not a single framework.** `langchain.agents.create_agent` by default;
  raw LangGraph (`langgraph-supervisor` / `langgraph-swarm`) only when a pattern needs it;
  `deepagents` reserved for long-horizon, planning-heavy work. See the ADR for the full rule.
- **Full SDLC, not a script.** One `uv` workspace, Ruff, mypy `--strict`, pytest split into
  `unit` / `integration` / `eval` markers, and GitHub Actions gating every PR.
- **Real infra alongside the agents.** Postgres (checkpointing/memory), self-hosted MLflow
  (observability + AI Gateway model access), and Milvus + Attu (vector store) — one
  `make up` away, not mocked out.

## Stack at a glance

- **Language:** Python 3.12, one **uv workspace** monorepo (`agents/*`, `packages/*`).
- **Agent frameworks:** `langchain.agents.create_agent` by default; raw LangGraph
  (`langgraph-supervisor` / `langgraph-swarm`) for multi-agent patterns; `deepagents` reserved
  for long-horizon, planning-heavy work. See the ADR for the decision rule.
- **Checkpointing / memory:** Postgres (`langgraph-checkpoint-postgres`) — `PostgresSaver` for
  per-thread state, `PostgresStore` for cross-thread memory. Browse it with pgAdmin
  (`infra/postgres/servers.json`), same "give the dev a UI" reasoning as Attu for Milvus.
- **Model access:** the self-hosted MLflow AI Gateway (`agents_common.get_chat_model`) — every
  agent calls a named gateway route via `ChatOpenAI` pointed at the gateway's OpenAI-compatible
  `/gateway/mlflow/v1` base URL, rather than holding a provider API key directly. The route
  itself is provisioned once via `make provision-gateway` against a self-hosted OpenAI-compatible
  model — see [`packages/mlflow-server/scripts/provision_gateway_route.py`](packages/mlflow-server/scripts/provision_gateway_route.py).
- **Observability:** self-hosted MLflow (`packages/mlflow-server`) — tracing via
  `mlflow.langchain.autolog()`, per-agent experiment tracking, and `mlflow.genai.evaluate()`
  for eval suites.
- **Vector store:** self-hosted Milvus standalone + Attu (`packages/milvus`) — `Settings.milvus_uri`
  is wired up for a future RAG-pattern agent; no example agent uses it yet.
- **Lint/format:** Ruff. **Types:** mypy (CI) + pyright (editor). **Tests:** pytest +
  Hypothesis, split into `unit` / `integration` / `eval` markers.
- **CI/CD:** GitHub Actions — `ci.yml` (lint/type/unit, every PR), `integration.yml`
  (Postgres-backed tests, every PR), `eval.yml` (MLflow eval suite, nightly/manual/labelled —
  not blocking, since it spends tokens), `ai-issue-discovery.yml` (MLflow AI Issue Discovery,
  weekly/manual — same non-blocking posture).
- **Containers:** root `docker-compose.yml` — Postgres + pgAdmin, MLflow, Milvus + Attu, and
  each agent as its own service/image.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Patterns and architecture](#patterns-and-architecture)
- [Repo layout](#repo-layout)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/), plus
Docker (Compose v2) for Postgres, MLflow, and Milvus.

```bash
make sync   # uv sync --all-packages — every agent + shared package, one workspace venv, one lockfile
```

See the [Makefile](Makefile) for every other available command; the sections below use it
throughout instead of the raw `uv`/`docker compose` invocations.

## Quick Start

```bash
# 1. Configure secrets — fill in MLFLOW_TRACKING_TOKEN (if your server has auth enabled) and
#    the SELFHOSTED_MODEL_* values for the model your gateway route should call
cp .env.example .env

# 2. Start the core stack: Postgres + pgAdmin, MLflow (+ its MCP server), and Milvus + Attu
#    (+ its MCP server) — also runs every provision-* step below automatically, including
#    registering mlflow-mcp/milvus-mcp/docs-langchain/reference-langchain in MLflow's MCP
#    Registry (see "MCP servers via MLflow's MCP Registry" below)
make up
open http://localhost:5050   # pgAdmin — the postgres service is pre-registered (servers.json),
                              # just enter the password from POSTGRES_PASSWORD when prompted
open http://localhost:5000   # MLflow UI
open http://localhost:3000   # Attu (Milvus UI) — no example agent uses Milvus yet, but it's
                              # running so you can poke at it; see packages/milvus/README.md

# 3. One-off: provision the MLflow AI Gateway route agent code calls (see .env.example's
#    SELFHOSTED_MODEL_* / GATEWAY_ROUTE_NAME vars). Only needs to be re-run if you delete the
#    mlflow-artifacts/mlflow-server Postgres data or change the route name.
make provision-gateway

# 4. One-off: sync eval dataset seed files (packages/mlflow-server/datasets/*.jsonl) into
#    MLflow's dataset registry — needed before `make test-eval`. Safe to re-run any time you
#    edit a dataset JSONL.
make provision-datasets

# 5. One-off: register + start each agent's production quality monitors (PRODUCTION_SCORERS in
#    its graph.py), so a sampled slice of live traces gets scored continuously — the "monitor"
#    half of MLflow GenAI eval-monitor, complementing `make test-eval`'s fixed-dataset CI run.
#    Safe to re-run any time you edit a PRODUCTION_SCORERS entry.
make provision-monitors

# 6. Run the reference agent
make demo
```

`make demo` runs `agents/patterns/react-agent` end to end against whatever model your MLflow AI Gateway
route resolves to, and prints its structured `AgentResponse` (`answer` + `used_tools`) — the
exact text depends on the model behind your gateway route, so it isn't reproduced here.

## Usage

### Run the test suites

```bash
make test-unit          # fast, no external services
make test-integration   # needs `make up` (real Postgres)
make test-eval           # needs `make provision-datasets` to have been run once; calls a real
                          # model via the gateway
make test                # unit + integration together
```

### Production quality monitoring

`make test-eval` scores a fixed dataset in CI. For quality signal on *live* traffic, each agent
also defines a `PRODUCTION_SCORERS` constant in its `graph.py` — scorers that run continuously
against a sampled slice of real production traces once registered:

```bash
make provision-monitors   # registers + starts every agent's PRODUCTION_SCORERS; idempotent
```

See [`agents_common.observability.register_production_monitors`](packages/agents-common/src/agents_common/observability/__init__.py)
and [`provision_monitors.py`](packages/mlflow-server/scripts/provision_monitors.py). Results show
up under each agent's experiment in the MLflow UI as trace assessments, not as a pass/fail
eval run.

### AI issue discovery

`PRODUCTION_SCORERS` tells you *whether* quality regressed against a fixed set of guidelines.
[MLflow AI Issue Discovery](https://mlflow.org/docs/latest/genai/eval-monitor/ai-insights/ai-issue-discovery/)
answers *why*: hypothesis-driven analysis over an experiment's existing traces that surfaces
operational issues (errors, timeouts, latency), quality issues (verbosity, inconsistency, bad
formatting) and success patterns, with example trace IDs and prioritized recommendations —
root-cause analysis over trace history, not continuous scoring. Implemented as
[`agents/examples/experiment-analysis-agent`](agents/examples/experiment-analysis-agent) — this
repo's first **Tier 3 (`deepagents`)** pattern, since the batched, hypothesis-driven analysis
(search traces, form/refine hypotheses over successive batches, write a report) is exactly the
long-horizon, planning-heavy shape that tier exists for.

Needs traces already in the target experiment (run `make demo`/`make demo-all` and/or let
`make provision-monitors` run for a while first). Two ways to run it, both on the same
self-hosted MLflow AI Gateway model every other agent in this repo uses — no separate API key:

```bash
make analyze-experiment EXPERIMENT=react-agent   # writes report-react-agent.md
```

- **Local, on demand**: `make analyze-experiment` as above (or
  `uv run --package experiment-analysis-agent experiment-analysis-agent <experiment>` directly).
- **Scheduled**: [`.github/workflows/ai-issue-discovery.yml`](.github/workflows/ai-issue-discovery.yml)
  runs the same command weekly (and on `workflow_dispatch`) for every implemented agent,
  uploading each report as a workflow artifact. Needs only the `MLFLOW_TRACKING_URI`/
  `MLFLOW_TRACKING_TOKEN` secrets `eval.yml` already uses.

Reach for this after a `PRODUCTION_SCORERS` metric regresses and you need to know why, before
writing a new eval dataset case, or just read the weekly scheduled report as a standing
spot-check. Like `eval.yml`, it's not a required PR check — it spends tokens and its output is
for a human to read, not a pass/fail signal.

#### MCP servers via MLflow's MCP Registry

`experiment_analysis_agent.graph` resolves its `mlflow-mcp` connection dynamically through
[MLflow's MCP Registry](https://mlflow.org/docs/latest/genai/mcp-registry/) — `.mcp.json`'s
`mlflow-mcp`/`milvus-mcp`/`docs-langchain`/`reference-langchain` entries are all registered
there too (`make provision-mcp-registry`, which `make up` runs automatically), so any agent can
look one up via `mlflow.genai.search_mcp_access_endpoints` instead of hardcoding a launch
command. `mlflow-mcp` and `milvus-mcp` needed converting from their stdio-only/localhost-only
CLIs into persistent streamable-http services first (`mlflow-mcp` and `milvus-mcp`
docker-compose services, `packages/mlflow-server/mlflow_mcp_server.py` /
`packages/milvus/milvus_mcp_server.py`) — MLflow's `create_mcp_access_endpoint` only accepts
`transport_type="streamable-http"`/`"sse"`, confirmed against the installed package. See
[`agents_common.mcp_servers`](packages/agents-common/src/agents_common/mcp_servers/__init__.py)
and `docs/decisions/0001-tech-stack.md` for the full investigation.

### Run everything containerized

```bash
make up-agents   # docker compose --profile agents up --build
```

Or run a single agent directly on the host instead of through Docker — see
[`agents/patterns/react-agent/README.md`](agents/patterns/react-agent/README.md) (and its equivalent for any
pattern you add) for the exact command.

## Configuration

Copy `.env.example` to `.env` and fill in the `SELFHOSTED_MODEL_*` values (plus
`MLFLOW_TRACKING_TOKEN`, if your server has auth enabled) — every other value has a working
local default. `agents_common.config.Settings` reads every agent-facing variable below; see
[docs/decisions/0001-tech-stack.md](docs/decisions/0001-tech-stack.md) for why each service is
configured this way.

| Variable | Default | Required | Description |
|---|---|---|---|
| `POSTGRES_USER` | `agents` | No | Role used for checkpointing/memory and (indirectly) MLflow's backend store |
| `POSTGRES_PASSWORD` | `change-me` | No | Change before running anywhere but a laptop |
| `POSTGRES_DB` | `agents` | No | Database agents checkpoint/store into — see `infra/postgres/init.sql` for the second (`mlflow`) database |
| `POSTGRES_HOST` | `localhost` | No | Overridden to `postgres` for containerized services |
| `POSTGRES_PORT` | `5432` | No | — |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | No | Overridden to `http://mlflow:5000` for containerized services |
| `MLFLOW_TRACKING_TOKEN` | *(empty)* | No | Authenticates tracking, evals, and calls *to* the gateway — only needed if your mlflow-server has auth enabled |
| `SELFHOSTED_MODEL_BASE_URL` | *(empty)* | **Yes** | Base URL of the self-hosted OpenAI-compatible model the gateway route should call. Only read by `make provision-gateway`, not by agent code |
| `SELFHOSTED_MODEL_API_KEY` | *(empty)* | **Yes** | Credential the *gateway* uses to call that model — distinct from `MLFLOW_TRACKING_TOKEN`. Only read by `make provision-gateway` |
| `SELFHOSTED_MODEL_NAME` | `gpt-oss-120b` | No | Model name sent to the self-hosted endpoint. Only read by `make provision-gateway` |
| `GATEWAY_ROUTE_NAME` | `gpt-oss-120b` | No | Gateway route name agent code calls via `get_chat_model(...)` — must match `react_agent.GATEWAY_ROUTE`. Only read by `make provision-gateway` |
| `MILVUS_URI` | `http://localhost:19530` | No | No example agent uses this yet — see `packages/milvus/README.md` |
| `LOG_LEVEL` | `INFO` | No | — |

pgAdmin's own login (`admin@example-agents.dev` / `admin`) is hardcoded in `docker-compose.yml`
rather than read from `.env` — it authenticates the pgAdmin UI itself, not Postgres, so it isn't
a secret worth plumbing through config (same treatment as MinIO's `minioadmin`/`minioadmin`
behind Milvus). Change it directly in `docker-compose.yml` if this ever runs somewhere
non-local.

## Patterns and architecture

| Pattern | Framework tier | Status | Path |
|---|---|---|---|
| Single ReAct agent | Tier 1 — `create_agent` | **Implemented** | `agents/patterns/react-agent` |
| Prompt Chaining | workflow — raw `StateGraph` | Stub | `agents/patterns/prompt-chaining-agent` |
| Routing | workflow — raw `StateGraph` | **Implemented** | `agents/patterns/routing-agent` |
| Parallelization | workflow — raw `StateGraph` | Stub | `agents/patterns/parallelization-agent` |
| Orchestrator-Workers | workflow — raw `StateGraph` | Stub | `agents/patterns/orchestrator-workers-agent` |
| Evaluator-Optimizer | workflow — raw `StateGraph` | Stub | `agents/patterns/evaluator-optimizer-agent` |
| Map-Reduce | workflow — raw `StateGraph` | Stub | `agents/patterns/map-reduce-agent` |
| Supervisor multi-agent | Tier 2 — LangGraph + `langgraph-supervisor` | Stub | `agents/patterns/supervisor-agent` |
| Swarm multi-agent | Tier 2 — LangGraph + `langgraph-swarm` | Stub | `agents/patterns/swarm-agent` |
| Network / Mesh | workflow — raw `StateGraph` | Stub | `agents/patterns/network-mesh-agent` |

Every pattern in this table has a full writeup under
[`docs/patterns/agent/`](docs/patterns/agent/) — see
**[agents/patterns/README.md](agents/patterns/README.md)** for the doc-to-package index. See
**[docs/decisions/0001-tech-stack.md](docs/decisions/0001-tech-stack.md)** for the full
reasoning behind the tiering and every other stack choice — this README is the quick-start,
that doc is the "why."

Pattern-specific content (how a given agent works, what it demonstrates, how to run or test it
directly on the host) lives in that agent's own README, not here — see
[`agents/patterns/react-agent/README.md`](agents/patterns/react-agent/README.md) for the one implemented pattern
today.

## Repo layout

```
example-agents/
├── docs/decisions/0001-tech-stack.md   # the ADR — read this first
├── docker-compose.yml                  # postgres/pgadmin + mlflow + milvus/attu + agent services
├── agents/
│   ├── patterns/                       # reusable pattern implementations
│   │   ├── react-agent/                # tier 1 — implemented
│   │   ├── supervisor-agent/           # tier 2 — stub
│   │   ├── swarm-agent/                # tier 2 — stub
│   │   └── ...                         # 7 more workflow-pattern stubs
│   └── examples/                       # applied demos composing patterns above
│       └── experiment-analysis-agent/  # tier 3 (deepagents) — MLflow AI Issue Discovery
├── packages/
│   ├── agents-common/                  # shared checkpointing/observability/config/mcp_servers
│   ├── mlflow-server/                  # self-hosted MLflow tracking server image
│   │   ├── mlflow_mcp_server.py         # mlflow-mcp over streamable-http (its own compose service)
│   │   ├── scripts/                    # provision_gateway_route.py, provision_datasets.py,
│   │   │                               # provision_monitors.py, provision_mcp_registry.py
│   │   └── datasets/                   # git-tracked eval dataset seed JSONL, one per agent
│   └── milvus/                         # self-hosted Milvus standalone + Attu (compose wiring)
│       └── milvus_mcp_server.py         # milvus-mcp over streamable-http (its own compose service)
└── infra/postgres/
    ├── init.sql                        # creates the second (mlflow) logical database
    └── servers.json                    # pre-registers postgres with pgAdmin on first boot
```

## Roadmap

- [x] Single ReAct agent (`agents/patterns/react-agent`)
- [ ] Supervisor multi-agent (`agents/patterns/supervisor-agent`)
- [ ] Swarm multi-agent (`agents/patterns/swarm-agent`)
- [ ] Remaining workflow-pattern stubs (`agents/patterns/{prompt-chaining,routing,parallelization,
      orchestrator-workers,evaluator-optimizer,map-reduce,network-mesh}-agent`)
- [ ] First RAG-pattern agent using `packages/milvus` — `Settings.milvus_uri` is already wired
      up in `agents-common`, waiting on an agent to use it
- [x] First Tier 3 (`deepagents`) usage — `agents/examples/experiment-analysis-agent`, backing
      `make analyze-experiment` / MLflow AI Issue Discovery

## Contributing

Contributions are welcome — this repo grows by adding one well-tested reference pattern at a
time.

1. Pick a tier from the table above and be able to justify it in one sentence (see the ADR's
   framework-tiering decision).
2. `uv init --lib --python 3.12 agents/patterns/<name>-agent` (or copy `agents/patterns/react-agent`'s layout).
3. Depend on `agents-common` for checkpointing/observability/config — don't re-implement it.
4. Ship `tests/unit`, `tests/integration`, and `tests/evals` from day one, same as
   `react-agent`.
5. Add the agent to `docker-compose.yml` under the `agents` profile if it should run
   standalone.

Before opening a PR:

```bash
make lint typecheck test-unit
```

`ci.yml` and `integration.yml` block merges on every PR; `eval.yml` runs the MLflow eval suite
separately (nightly, on demand, or on a PR labelled `run-evals`) since it costs tokens — see
the ADR's CI/CD section for why they're split.

## License

[MIT](LICENSE) © 2026 Joseph Searle
