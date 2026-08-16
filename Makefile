.PHONY: sync lint format typecheck test test-unit test-integration test-eval up up-agents down reset logs demo provision-gateway provision-datasets provision-prompts

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
	docker compose up -d --wait postgres pgadmin mlflow milvus-etcd milvus-minio milvus-standalone attu
	$(MAKE) provision-prompts

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
