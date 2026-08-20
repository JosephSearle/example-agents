"""Provision Milvus collections from the git-tracked seed JSONL files.

Reads packages/milvus/collections/. Each <collection-name>.jsonl file becomes a Milvus collection
named after the file stem, with hyphens replaced by underscores (Milvus collection names must
match `^[a-zA-Z_][a-zA-Z0-9_]*$`) — e.g. basic-rag-agent.jsonl -> collection `basic_rag_agent`,
matching `basic_rag_agent.graph.COLLECTION_NAME`.

Unlike packages/mlflow-server/scripts/provision_prompts.py's diff-then-skip idempotency (worth
preserving prompt version history), these are demo/test seed collections, not versioned
production data — idempotency here is drop-if-exists-then-recreate: simpler, and it guarantees
every re-run reflects the current seed file exactly rather than accumulating stale rows from a
previous schema/embedding-model.

Depends on an embeddings-capable MLflow AI Gateway route already being provisioned (see
packages/mlflow-server/scripts/provision_gateway_route.py's embeddings-route provisioning) and on
`langchain-milvus`/`pymilvus` being importable — those are basic-rag-agent's own dependencies
(packages/milvus has no pyproject.toml of its own, see its README), so `basic-rag-agent` must be
synced into the workspace venv for this script to run.

Run after `make up` (needs a live milvus-standalone + an embeddings gateway route); wired into
`make up` itself via the `provision-milvus-collections` target:

    uv run python packages/milvus/scripts/provision_collections.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents_common import get_embeddings, get_settings
from agents_common.logging import configure_logging
from langchain_milvus import Milvus
from pymilvus import MilvusClient  # type: ignore[import-untyped]
import structlog

COLLECTIONS_DIR = Path(__file__).resolve().parent.parent / "collections"

# Must match basic_rag_agent.graph.EMBEDDING_GATEWAY_ROUTE — see that constant's own comment for
# why this is a second, separately-provisioned gateway route from the chat route every other
# agent in this repo calls.
_EMBEDDING_GATEWAY_ROUTE = "text-embedding"

_logger = structlog.get_logger(__name__)


def _collection_name(jsonl_path: Path) -> str:
    return jsonl_path.stem.replace("-", "_")


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _provision_one(jsonl_path: Path, *, milvus_uri: str) -> None:
    collection_name = _collection_name(jsonl_path)
    records = _load_records(jsonl_path)

    client = MilvusClient(uri=milvus_uri)
    if client.has_collection(collection_name):
        _logger.info("dropping_existing_collection", collection=collection_name)
        client.drop_collection(collection_name)
    client.close()

    _logger.info(
        "provisioning_collection",
        collection=collection_name,
        source=jsonl_path.name,
        record_count=len(records),
    )
    Milvus.from_texts(
        texts=[record["text"] for record in records],
        embedding=get_embeddings(_EMBEDDING_GATEWAY_ROUTE),
        metadatas=[record["metadata"] for record in records],
        collection_name=collection_name,
        connection_args={"uri": milvus_uri},
    )
    _logger.info("collection_provisioned", collection=collection_name, record_count=len(records))


def main() -> None:
    """Sync every seed JSONL in COLLECTIONS_DIR into a Milvus collection."""
    configure_logging()
    milvus_uri = get_settings().milvus_uri

    for jsonl_path in sorted(COLLECTIONS_DIR.glob("*.jsonl")):
        _provision_one(jsonl_path, milvus_uri=milvus_uri)

    _logger.info("done")


if __name__ == "__main__":
    main()
