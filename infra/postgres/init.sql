-- Runs once, on first container start (docker-entrypoint-initdb.d convention).
-- POSTGRES_DB (from .env, default "agents") is created automatically by the postgres image;
-- this adds the second logical database used by the MLflow tracking server's backend store,
-- kept separate from agent checkpoint/store data per docs/decisions/0001-tech-stack.md.
CREATE DATABASE mlflow;
