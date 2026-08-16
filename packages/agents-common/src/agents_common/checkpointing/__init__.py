"""Postgres-backed checkpointing (short-term) and store (long-term memory) factories.

Every agent pattern in this repo — ReAct, supervisor, swarm, deep agent — resumes from the
same durable state: `PostgresSaver` checkpoints the graph's state per-thread (a conversation),
and `PostgresStore` holds cross-thread memories (facts that outlive a single conversation).
Centralising this here means a connection-pool or schema change happens once.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from agents_common.config import get_settings

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def get_checkpointer(postgres_uri: str | None = None) -> Iterator[PostgresSaver]:
    """Yield a `PostgresSaver`, running `.setup()` idempotently on first use.

    Args:
        postgres_uri: Override the connection string; defaults to `Settings.postgres_uri`.

    Yields:
        A `PostgresSaver` ready to pass as `checkpointer=` to `create_agent` / `StateGraph.compile`.
    """
    uri = postgres_uri or get_settings().postgres_uri
    with PostgresSaver.from_conn_string(uri) as checkpointer:
        checkpointer.setup()
        yield checkpointer


@contextmanager
def get_store(postgres_uri: str | None = None) -> Iterator[PostgresStore]:
    """Yield a `PostgresStore` for long-term, cross-thread memory.

    Args:
        postgres_uri: Override the connection string; defaults to `Settings.postgres_uri`.

    Yields:
        A `PostgresStore` ready to pass as `store=` to `create_agent` / `StateGraph.compile`.
    """
    uri = postgres_uri or get_settings().postgres_uri
    with PostgresStore.from_conn_string(uri) as store:
        store.setup()
        yield store
