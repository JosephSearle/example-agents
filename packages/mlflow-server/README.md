# mlflow-server

The self-hosted MLflow tracking server as its own image, separate from any agent — built per
https://mlflow.org/docs/latest/self-hosting/, with two changes for this repo:

1. **Postgres backend store** instead of the SQLite default, so concurrent writers (multiple
   agent processes tracing at once) don't corrupt the tracking DB.
2. **Bind-mounted local artifact store** in dev (`/mlartifacts`, see `docker-compose.yml`);
   swap `MLFLOW_ARTIFACT_ROOT` for an S3/GCS/Azure Blob URI in a real deployment — the image
   itself doesn't change.

Built and run via the root `docker-compose.yml`:

```bash
docker compose up -d postgres mlflow
open http://localhost:5000
```

## Provisioning

Two kinds of agent-owned content are checked into this package and synced into MLflow's
registries by scripts in `scripts/`:

- **`datasets/<agent-name>.jsonl`** — eval datasets, synced via `scripts/provision_datasets.py`
  (`make provision-datasets`, manual).
- **`prompts/<agent-name>.txt`** — system prompts, synced via `scripts/provision_prompts.py`
  into MLflow's prompt registry, with a `production` alias pointed at the current version.
  Agents load their prompt at runtime via `mlflow.genai.load_prompt("prompts:/<agent-name>@production")`
  instead of importing it from a Python module — see
  `agents/patterns/react-agent/src/react_agent/graph.py`'s `load_system_prompt`. Unlike
  datasets and the gateway route, prompt provisioning runs **automatically** as part of
  `make up` (`provision-prompts` is chained after the compose services report healthy), so a
  fresh `make up` always leaves `production`-aliased prompts in sync with the checked-in files.
  It's also idempotent — re-running with unchanged files registers no new version.

`scripts/provision_gateway_route.py` (`make provision-gateway`, manual) provisions the MLflow AI
Gateway route agents call for model access — not agent-owned content, but the same "one-off
setup script run against a live mlflow-server" shape.
