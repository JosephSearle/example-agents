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

from langchain_openai import ChatOpenAI

from agents_common.config import Settings, get_settings


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
        api_key=settings.mlflow_tracking_token or "unused",  # type: ignore[arg-type]
        model=gateway_route,
        base_url=settings.mlflow_gateway_base_url,
        # langchain_openai only auto-enables stream_usage when the client uses the default
        # OpenAI base URL — we always pass the gateway's custom base_url, which disables that
        # auto-enable. Without it, streamed responses never carry usage_metadata, so MLflow's
        # autolog tracer has nothing to extract into mlflow.chat.tokenUsage.
        stream_usage=True,
    )
