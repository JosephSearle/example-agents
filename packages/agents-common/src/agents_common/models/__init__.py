"""Chat model access via the self-hosted MLflow AI Gateway.

Every agent in this repo gets its chat model through a named route on the self-hosted MLflow
AI Gateway rather than holding a provider API key directly. Which model actually answers a
given route (a self-hosted vLLM/TGI deployment, etc.) is a gateway-side config concern —
agent code only ever names a route, e.g. `get_chat_model("gpt-oss-120b")`. The one credential
this repo needs is `MLFLOW_TRACKING_TOKEN`: the same bearer token authenticates tracking,
`mlflow.genai.evaluate()`, and gateway model calls alike, since they're all MLflow REST client
calls under the hood.

NOTE: The gateway is mounted directly on the MLflow tracking server (mlflow[genai]>=3.1) at
`/gateway/mlflow/v1`, and it's OpenAI Chat Completions-compatible — so `ChatOpenAI` pointed at
that base_url is the client, not a MLflow-specific LangChain integration. See
`Settings.mlflow_gateway_base_url`. Routes are provisioned via the gateway's REST API (secret →
model definition → endpoint); see `packages/mlflow-server/scripts/provision_gateway_route.py`.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
import requests

from agents_common.config import Settings, get_settings

# Placeholder passed to a client's api_key when the gateway/reranker has no auth enabled — several
# HTTP clients in this module reject an empty-string/None api_key at construction time, and this
# same convention is used gateway-side for its own upstream secret in provision_gateway_route.py.
# Centralised so every "is auth actually enabled" check agrees on what "disabled" looks like.
_UNAUTHENTICATED_PLACEHOLDER = "unused"


def _bearer_headers(api_key: str) -> dict[str, str]:
    """Build an Authorization header, or none, from a possibly-placeholder api_key."""
    if api_key and api_key != _UNAUTHENTICATED_PLACEHOLDER:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def get_chat_model(gateway_route: str, *, settings: Settings | None = None) -> ChatOpenAI:
    """Build a chat model bound to a named route on the self-hosted MLflow AI Gateway.

    Args:
        gateway_route: The AI Gateway route/endpoint name to call, e.g. "gpt-oss-120b".
            Routes are provisioned gateway-side; define the route name your agent uses as a
            constant in that agent's own package (see `react_agent.GATEWAY_ROUTE`), the same
            way each agent owns its `EXPERIMENT_NAME`.
        settings: Override settings; defaults to `get_settings()`.

    Returns:
        A `ChatOpenAI` instance ready to pass as `model=` to `create_agent` or bind into a
        LangGraph node.
    """
    settings = settings or get_settings()
    return ChatOpenAI(
        # openai-python rejects an empty-string api_key at construction time (raises
        # OpenAIError before any request is made) — fall back to a placeholder when the
        # gateway has no auth enabled, same convention used for the gateway's own upstream
        # secret in provision_gateway_route.py.
        api_key=settings.mlflow_tracking_token or _UNAUTHENTICATED_PLACEHOLDER,  # type: ignore[arg-type]
        model=gateway_route,
        base_url=settings.mlflow_gateway_base_url,
        # langchain_openai only auto-enables stream_usage when the client uses the default
        # OpenAI base URL — we always pass the gateway's custom base_url, which disables that
        # auto-enable. Without it, streamed responses never carry usage_metadata, so MLflow's
        # autolog tracer has nothing to extract into mlflow.chat.tokenUsage.
        stream_usage=True,
        # The gateway's OpenAI-compatible endpoint only implements the (legacy) Chat
        # Completions API, never the Responses API — but langchain-openai 1.0 defaults
        # `output_version` to "responses/v1" and infers which API to call partly from the
        # model name, independent of base_url. Left implicit, some message shapes (e.g. a deep
        # agent's assembled multi-block system prompt) get serialized as Responses-API-style
        # content blocks, which the gateway's stricter Chat Completions schema then rejects
        # with a 400 ("content.0 ... Input should be a valid dictionary"). Pin both explicitly
        # so every agent in this repo always targets Chat Completions with plain-string
        # content, matching what the gateway actually speaks — see langchain-openai's docs on
        # "Model name can trigger Responses API routing" for OpenAI-compatible providers.
        use_responses_api=False,
        output_version="v0",
    )


class _GatewayEmbeddings(Embeddings):
    """Embeddings client for a named route on the self-hosted MLflow AI Gateway.

    Unlike chat, the gateway has no generic OpenAI-compatible `/v1/embeddings` path — only
    `/gateway/mlflow/v1/chat/completions` is registered as a fixed route (see
    `mlflow.server.gateway_api.chat_completions`, which `Settings.mlflow_gateway_base_url` +
    `get_chat_model`'s `ChatOpenAI` client target). Embeddings only exist per-route, via the
    unified `POST /gateway/{endpoint_name}/mlflow/invocations` handler (see
    `mlflow.server.gateway_api.invocations`, mounted directly under `/gateway`, *not* under
    `/gateway/mlflow/v1` — a sibling path, not a suffix of `mlflow_gateway_base_url`) — it
    detects chat vs. embeddings from the payload shape (`"messages"` vs `"input"`). So this
    can't reuse `OpenAIEmbeddings`, whose client always posts to a fixed `{base_url}/embeddings`
    suffix, nor `mlflow_gateway_base_url` as its base. This instead posts directly to that
    route's `invocations` URL (built from `Settings.mlflow_tracking_uri`) with
    `{"input": [...]}`, matching `mlflow.gateway.schemas.embeddings.RequestPayload`/
    `ResponsePayload` — the same shape as an OpenAI embeddings response (`{"data": [{"embedding":
    [...], "index": ...}, ...]}`).
    """

    def __init__(self, *, gateway_route: str, tracking_uri: str, api_key: str) -> None:
        self._invocations_url = f"{tracking_uri}/gateway/{gateway_route}/mlflow/invocations"
        self._session = requests.Session()
        self._session.headers.update(_bearer_headers(api_key))

    def _embed(self, input_: list[str]) -> list[list[float]]:
        response = self._session.post(self._invocations_url, json={"input": input_}, timeout=60)
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def get_embeddings(gateway_route: str, *, settings: Settings | None = None) -> Embeddings:
    """Build an embeddings model bound to a named route on the self-hosted MLflow AI Gateway.

    Same "route name only, gateway owns the upstream" contract as `get_chat_model`, since the
    gateway is the only model access path this repo uses (see this module's docstring). The
    route itself must be provisioned separately, backed by an embeddings-capable upstream — see
    `packages/mlflow-server/scripts/provision_gateway_route.py`'s embeddings-route provisioning.

    Args:
        gateway_route: The AI Gateway route/endpoint name to call, e.g. "text-embedding". Define
            the route name your agent uses as a constant in that agent's own package (see
            `basic_rag_agent.EMBEDDING_GATEWAY_ROUTE`), the same way each agent owns its chat
            `GATEWAY_ROUTE`.
        settings: Override settings; defaults to `get_settings()`.

    Returns:
        An `Embeddings` instance ready to pass as `embedding_function=` to
        `langchain_milvus.Milvus`.
    """
    settings = settings or get_settings()
    return _GatewayEmbeddings(
        gateway_route=gateway_route,
        tracking_uri=settings.mlflow_tracking_uri,
        api_key=settings.mlflow_tracking_token or _UNAUTHENTICATED_PLACEHOLDER,
    )


class RerankResult(TypedDict):
    """One reranked document — `index` refers back into the `documents` list passed to `rerank`."""

    index: int
    score: float


def _parse_rerank_response(payload: dict[str, Any]) -> list[RerankResult]:
    """Isolate the reranker response shape so a wrong assumption is a one-function fix.

    See `_Reranker`'s docstring for why this shape is assumed rather than confirmed.
    """
    return sorted(
        ({"index": item["index"], "score": item["relevance_score"]} for item in payload["results"]),
        key=lambda result: result["score"],
        reverse=True,
    )


class _Reranker:
    """HTTP client for a self-hosted cross-encoder reranker, called directly.

    Not through the MLflow AI Gateway. A cross-encoder reranker isn't chat- or embeddings-shaped,
    so it doesn't fit the gateway's secret/model-definition/endpoint provisioning at all (see
    `packages/mlflow-server/scripts/provision_gateway_route.py` and `_GatewayEmbeddings`'s
    docstring for what the gateway *does* model). This is instead a plain, directly-configured
    HTTP endpoint — same treatment `Settings.milvus_uri` already gets — read from
    `RERANKER_MODEL_BASE_URL`.

    Confirmed live (not TEI, despite matching this repo's embeddings model's OpenShift/BGE
    deployment pattern): a vLLM/OpenAI-compatible rerank endpoint, `POST /rerank` with
    `{"model": str, "query": str, "documents": [str]}` -> `{"results": [{"index": int,
    "relevance_score": float, ...}], ...}`.
    """

    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self._rerank_url = f"{base_url.rstrip('/')}/rerank"
        self._model = model
        self._session = requests.Session()
        self._session.headers.update(_bearer_headers(api_key))

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[RerankResult]:
        response = self._session.post(
            self._rerank_url,
            json={"model": self._model, "query": query, "documents": documents},
            timeout=30,
        )
        response.raise_for_status()
        return _parse_rerank_response(response.json())[:top_n]


def get_reranker(*, settings: Settings | None = None) -> _Reranker:
    """Build a reranker client bound to `Settings.reranker_model_base_url`.

    See `_Reranker`'s docstring for why this bypasses the MLflow AI Gateway entirely, unlike
    `get_chat_model`/`get_embeddings`.

    Args:
        settings: Override settings; defaults to `get_settings()`.

    Returns:
        A `_Reranker` instance whose `.rerank(query, documents, *, top_n=...)` returns the
        top `top_n` documents (by index into `documents`) sorted by descending relevance score.
    """
    settings = settings or get_settings()
    return _Reranker(
        base_url=settings.reranker_model_base_url,
        api_key=settings.reranker_model_api_key or _UNAUTHENTICATED_PLACEHOLDER,
        model=settings.reranker_model_name,
    )
