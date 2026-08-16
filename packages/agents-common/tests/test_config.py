"""Unit tests for agents_common.config."""

from __future__ import annotations

from agents_common.config import Settings, get_settings
import pytest

pytestmark = pytest.mark.unit

_ENV_KEYS = (
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_TOKEN",
    "MILVUS_URI",
    "LOG_LEVEL",
)


def test_settings_defaults_when_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env vars set, Settings falls back to its documented dev defaults."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.log_level == "INFO"
    assert settings.mlflow_tracking_token == ""
    assert settings.postgres_uri == "postgresql://agents:change-me@localhost:5432/agents"
    assert settings.milvus_uri == "http://localhost:19530"


def test_postgres_uri_is_built_from_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """postgres_uri is derived from the individual POSTGRES_* values, not set directly."""
    monkeypatch.setenv("POSTGRES_USER", "svc")
    monkeypatch.setenv("POSTGRES_PASSWORD", "s3cr3t")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("POSTGRES_DB", "checkpoints")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.postgres_uri == "postgresql://svc:s3cr3t@db.internal:6543/checkpoints"


def test_settings_reads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings picks up overrides from environment variables by alias."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "test-token")
    monkeypatch.setenv("MILVUS_URI", "http://milvus-standalone:19530")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.log_level == "DEBUG"
    assert settings.mlflow_tracking_token == "test-token"
    assert settings.milvus_uri == "http://milvus-standalone:19530"


def test_get_settings_is_cached() -> None:
    """get_settings() returns the same instance across calls (lru_cache)."""
    assert get_settings() is get_settings()
