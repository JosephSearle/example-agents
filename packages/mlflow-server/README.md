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
