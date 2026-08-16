.PHONY: sync lint format typecheck test test-unit test-integration test-eval up up-agents down logs demo provision-gateway provision-datasets

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
	docker compose up -d postgres pgadmin mlflow milvus-etcd milvus-minio milvus-standalone attu

up-agents:
	docker compose --profile agents up --build

down:
	docker compose down

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
