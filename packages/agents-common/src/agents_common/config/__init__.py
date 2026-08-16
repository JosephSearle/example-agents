"""Typed settings shared by every agent, loaded from the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved relative to this file, not the process's cwd — a relative "​.env" would silently
# resolve against wherever the caller happened to `cd` first (e.g. `make demo`'s
# `cd agents/react-agent && ...`), missing the repo-root .env entirely and falling back to
# defaults instead of erroring, which is worse: it looks like a real (wrong) config rather than
# a missing file.
_REPO_ROOT_ENV_FILE = Path(__file__).resolve().parents[5] / ".env"


class Settings(BaseSettings):
    """Environment-driven configuration.

    Mirrors `.env.example` at the repo root. Every agent constructs this once at startup
    rather than reading `os.environ` directly, so config is typed, validated, and easy to
    override in tests.
    """

    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    # Individual Postgres components rather than one connection-string field: a dev who
    # rotates the password or moves to a managed instance edits one value here instead of
    # having to re-derive and re-paste a full DSN into a second place. `postgres_uri` below is
    # always derived from these, never set directly.
    postgres_user: str = Field(default="agents", alias="POSTGRES_USER")
    postgres_password: str = Field(default="change-me", alias="POSTGRES_PASSWORD")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="agents", alias="POSTGRES_DB")

    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    # The only credential this repo needs: every model call goes through the self-hosted
    # MLflow AI Gateway rather than a direct provider API key, and this one bearer token
    # authenticates tracking, evals, and gateway model calls alike. See
    # `agents_common.models.get_chat_model`.
    mlflow_tracking_token: str = Field(default="", alias="MLFLOW_TRACKING_TOKEN")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mlflow_gateway_base_url(self) -> str:
        """The MLflow AI Gateway's OpenAI-compatible base URL, derived from the tracking URI.

        The gateway is mounted on the tracking server itself at this fixed path (see
        `mlflow.server.gateway_api`) — not a separately configured value.
        """
        return f"{self.mlflow_tracking_uri}/gateway/mlflow/v1"

    # No agent uses Milvus yet — this exists so a future RAG-pattern agent gets the same "one
    # typed value, not a hand-assembled connection string" treatment as postgres_uri and
    # mlflow_tracking_uri, rather than inventing its own env var later. See packages/milvus.
    milvus_uri: str = Field(default="http://localhost:19530", alias="MILVUS_URI")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def postgres_uri(self) -> str:
        """The full Postgres DSN, built from the individual components above.

        Not settable directly — change `POSTGRES_HOST`/`POSTGRES_PORT`/etc. instead. Every
        caller (checkpointer, store, tests) reads this property rather than assembling its
        own connection string.
        """
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached `Settings` instance."""
    return Settings()
