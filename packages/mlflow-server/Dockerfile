# Self-hosted MLflow tracking server. Backed by Postgres (see docker-compose.yml) rather than
# the SQLite default, which isn't safe for concurrent writers — see
# docs/decisions/0001-tech-stack.md.
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 1000 mlflow && \
    mkdir -p /mlartifacts && chown mlflow:mlflow /mlartifacts
USER mlflow

EXPOSE 5000

# --backend-store-uri and --default-artifact-root are supplied at runtime via
# docker-compose.yml so the same image works in dev (local volume) and prod
# (swap MLFLOW_ARTIFACT_ROOT for an s3:// / gs:// / abfs:// URI).
ENTRYPOINT ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5000"]
