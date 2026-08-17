.PHONY: sync lint format typecheck test test-unit test-integration test-eval up up-agents down reset logs demo demo-all provision-gateway provision-datasets provision-prompts provision-monitors provision-mcp-registry analyze-experiment tekton-mcp-up tekton-mcp-down

sync:
	uv sync --all-packages

lint:
	uv run ruff check agents packages

format:
	uv run ruff format agents packages

typecheck:
	uv run mypy agents packages

test-unit:
	uv run pytest -m unit

test-integration:
	uv run pytest -m integration

test-eval:
	uv run pytest -m eval

test: test-unit test-integration

up:
	docker compose up -d --wait postgres pgadmin mlflow mlflow-mcp milvus-etcd milvus-minio milvus-standalone milvus-mcp atlassian-mcp attu
	$(MAKE) provision-gateway
	$(MAKE) provision-prompts
	$(MAKE) provision-datasets
	$(MAKE) provision-monitors
	$(MAKE) provision-mcp-registry

up-agents:
	docker compose --profile agents up --build

down:
	docker compose down

# Destructive: stops all services and deletes their volumes (Postgres data incl. the MLflow
# backend store, MLflow artifacts, Milvus data), then brings everything back up from scratch
# (services + gateway route not included — that's still a manual `make provision-gateway`).
# Use this when local state has drifted or you just want a genuinely clean slate.
reset:
	docker compose down -v
	$(MAKE) up

logs:
	docker compose logs -f

demo:
	cd agents/patterns/react-agent && uv run react-agent "What's 47 * 12, and does that number mean anything in dev slang?"

# Runs every fully-implemented pattern (react-agent, routing-agent, prompt-chaining-agent —
# evaluator-optimizer and the rest under agents/patterns/ are still stubs, see README's pattern
# table) concurrently as background jobs, each against its own MLflow experiment/thread, so their
# traces don't collide. `uv run --package <name> <entrypoint>` is used instead of `cd`ing into
# each agent's directory, since backgrounding `cd X && ...; cd Y && ...` from one shell would
# race; running from the repo root avoids that. Needs `make up` (+ `make provision-gateway` /
# `provision-prompts` / `provision-datasets` if not already run) first, same as `make demo`.
demo-all:
	uv run --package react-agent react-agent "What's 47 * 12, and does that number mean anything in dev slang?" & \
	uv run --package routing-agent routing-agent "My last payment was charged twice, can I get a refund?" & \
	uv run --package prompt-chaining-agent prompt-chaining-agent "The history of the semicolon in programming languages" & \
	wait

# One-off: provisions the MLflow AI Gateway route agent code calls (see .env.example's
# SELFHOSTED_MODEL_* / GATEWAY_ROUTE_NAME vars). Run once after `make up`, before running
# integration/eval tests or agents against the local mlflow container.
provision-gateway:
	uv run python packages/mlflow-server/scripts/provision_gateway_route.py

# One-off (idempotent, safe to re-run): syncs packages/mlflow-server/datasets/*.jsonl into
# MLflow's dataset registry, one dataset per agent, tagged to that agent's experiment. Run once
# after `make up`, before `make test-eval`. Re-run after editing a dataset JSONL to push the
# updated records into MLflow.
provision-datasets:
	uv run python packages/mlflow-server/scripts/provision_datasets.py

# Idempotent, safe to re-run: syncs packages/mlflow-server/prompts/*.txt into MLflow's prompt
# registry, one prompt per agent, aliased "production". Runs automatically as part of `make up`
# (needs a live mlflow-server, hence chained after `docker compose up --wait`); re-run directly
# after editing a prompt file to push the update without a full `make up`.
provision-prompts:
	uv run python packages/mlflow-server/scripts/provision_prompts.py

# One-off (idempotent, safe to re-run): registers + starts each agent's PRODUCTION_SCORERS
# (defined in its graph.py) so they score a sampled slice of live production traces, complementing
# `make test-eval`'s fixed-dataset CI run — see packages/mlflow-server/scripts/provision_monitors.py.
# Not run automatically by `make up`: turning monitoring on is a deliberate step, not part of
# bringing local infra up. Run after `make up` + `make provision-gateway`; re-run after editing a
# PRODUCTION_SCORERS entry to push the update.
provision-monitors:
	uv run python packages/mlflow-server/scripts/provision_monitors.py

# Idempotent, safe to re-run: registers every server in .mcp.json (mlflow-mcp, milvus-mcp,
# docs-langchain, reference-langchain) in MLflow's MCP Registry, creates each one's access
# endpoint, and refreshes its discovered-tools snapshot — see
# packages/mlflow-server/scripts/provision_mcp_registry.py and agents_common.mcp_servers, which
# resolves these at runtime. `--with 'mlflow[mcp]>=3.5.1'` is ephemeral (not a permanent
# dependency of any agent package) — only this script's tool-discovery step needs it. Runs
# automatically as part of `make up` (needs `mlflow`, `mlflow-mcp`, and `milvus-mcp` healthy,
# hence chained after `docker compose up --wait`) since experiment-analysis-agent depends on it
# to function at all, not an opt-in like `provision-monitors`.
provision-mcp-registry:
	uv run --with 'mlflow[mcp]>=3.5.1' python packages/mlflow-server/scripts/provision_mcp_registry.py

# tekton-mcp-server (TEKTON_MCP_BINARY) is a native macOS binary from a separate repo
# (github.com/tektoncd/mcp-server, not part of this uv workspace) — it can't be containerized
# here, so unlike atlassian-mcp/milvus-mcp/mlflow-mcp it isn't a compose service. This runs it
# directly on the host over HTTP instead, so provision-mcp-registry has something reachable to
# register — you start/stop it yourself, same host-reachable-URL caveat this repo already
# documents for the compose-managed MCP services. Override TEKTON_MCP_BINARY if yours lives
# somewhere other than the default below.
TEKTON_MCP_BINARY ?= /Users/josephsearle/Documents/Projects/mcp-server/bin/tekton-mcp-server
TEKTON_MCP_PORT ?= 8080
TEKTON_MCP_PIDFILE := /tmp/tekton-mcp-server.pid
tekton-mcp-up:
	@if [ -f $(TEKTON_MCP_PIDFILE) ] && kill -0 $$(cat $(TEKTON_MCP_PIDFILE)) 2>/dev/null; then \
		echo "tekton-mcp-server already running (pid $$(cat $(TEKTON_MCP_PIDFILE)))"; \
	else \
		$(TEKTON_MCP_BINARY) -transport http -address :$(TEKTON_MCP_PORT) & \
		echo $$! > $(TEKTON_MCP_PIDFILE); \
		echo "tekton-mcp-server started (pid $$!) on :$(TEKTON_MCP_PORT)"; \
	fi

tekton-mcp-down:
	@if [ -f $(TEKTON_MCP_PIDFILE) ]; then \
		kill $$(cat $(TEKTON_MCP_PIDFILE)) 2>/dev/null || true; \
		rm -f $(TEKTON_MCP_PIDFILE); \
		echo "tekton-mcp-server stopped"; \
	else \
		echo "tekton-mcp-server not running"; \
	fi

# Runs MLflow AI Issue Discovery against one agent's experiment via
# agents/examples/experiment-analysis-agent — this repo's Tier 3 (`deepagents`) pattern,
# producing a markdown report of operational/quality issues found in its traces. Complements
# `provision-monitors`'s continuous PRODUCTION_SCORERS with root-cause analysis over historical
# traces instead of live per-trace scoring. Needs traces to already exist in the experiment
# (run `make demo`/`make demo-all` and/or `make provision-monitors` first). EXPERIMENT defaults
# to react-agent; override e.g. `make analyze-experiment EXPERIMENT=routing-agent`. Runs on the
# same self-hosted MLflow AI Gateway model as every other agent (see
# experiment-analysis-agent's GATEWAY_ROUTE) — no separate API key needed, unattended or not,
# which is also what lets `ai-issue-discovery.yml` run this on a schedule without its own
# credential. See agents/examples/experiment-analysis-agent/README.md.
EXPERIMENT ?= react-agent
analyze-experiment:
	uv run --package experiment-analysis-agent experiment-analysis-agent $(EXPERIMENT)
