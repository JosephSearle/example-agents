"""Unit tests for agents_common.mcp_servers."""

from __future__ import annotations

from unittest.mock import MagicMock

from agents_common import mcp_servers
from agents_common.config import Settings
from agents_common.mcp_servers import milvus_mcp_connection, mlflow_mcp_connection
import pytest

pytestmark = pytest.mark.unit


def _settings(**overrides: str) -> Settings:
    values = {
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "MLFLOW_TRACKING_TOKEN": "test-token",
        "MILVUS_URI": "http://localhost:19530",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def _fake_endpoint(*, url: str, transport_type: str) -> MagicMock:
    endpoint = MagicMock()
    endpoint.url = url
    endpoint.transport_type.value = transport_type
    return endpoint


def test_mlflow_mcp_connection_resolves_streamable_http_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _fake_endpoint(url="http://localhost:8001/mcp", transport_type="streamable-http")
    monkeypatch.setattr(mcp_servers, "search_mcp_access_endpoints", lambda **_: [endpoint])
    monkeypatch.setattr(mcp_servers.mlflow, "set_tracking_uri", lambda _: None)

    connection = mlflow_mcp_connection(_settings())

    # MLflow's registry uses "streamable-http" (hyphen); langchain-mcp-adapters expects
    # "streamable_http" (underscore) — this is the translation, not a passthrough.
    assert connection["transport"] == "streamable_http"
    assert connection["url"] == "http://localhost:8001/mcp"


def test_mlflow_mcp_connection_maps_sse_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = _fake_endpoint(url="http://localhost:8001/mcp", transport_type="sse")
    monkeypatch.setattr(mcp_servers, "search_mcp_access_endpoints", lambda **_: [endpoint])
    monkeypatch.setattr(mcp_servers.mlflow, "set_tracking_uri", lambda _: None)

    connection = mlflow_mcp_connection(_settings())

    assert connection["transport"] == "sse"


def test_mlflow_mcp_connection_raises_when_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_servers, "search_mcp_access_endpoints", lambda **_: [])
    monkeypatch.setattr(mcp_servers.mlflow, "set_tracking_uri", lambda _: None)

    with pytest.raises(RuntimeError, match="provision-mcp-registry"):
        mlflow_mcp_connection(_settings())


def test_milvus_mcp_connection_resolves_streamable_http_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _fake_endpoint(url="http://localhost:8002/mcp", transport_type="streamable-http")
    monkeypatch.setattr(mcp_servers, "search_mcp_access_endpoints", lambda **_: [endpoint])
    monkeypatch.setattr(mcp_servers.mlflow, "set_tracking_uri", lambda _: None)

    connection = milvus_mcp_connection(_settings())

    assert connection["transport"] == "streamable_http"
    assert connection["url"] == "http://localhost:8002/mcp"


def test_milvus_mcp_connection_raises_when_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_servers, "search_mcp_access_endpoints", lambda **_: [])
    monkeypatch.setattr(mcp_servers.mlflow, "set_tracking_uri", lambda _: None)

    with pytest.raises(RuntimeError, match="provision-mcp-registry"):
        milvus_mcp_connection(_settings())
