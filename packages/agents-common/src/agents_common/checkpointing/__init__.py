"""Postgres-backed checkpointing (short-term) and store (long-term memory) factories.

Every agent pattern in this repo — ReAct, supervisor, swarm, deep agent — resumes from the
same durable state: `PostgresSaver` checkpoints the graph's state per-thread (a conversation),
and `PostgresStore` holds cross-thread memories (facts that outlive a single conversation).
Centralising this here means a connection-pool or schema change happens once.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
import psycopg

from agents_common.config import get_settings

if TYPE_CHECKING:
    from collections.abc import Iterator

# Arbitrary fixed keys for Postgres session-level advisory locks (pg_advisory_lock), one per
# `.setup()` call site below. `PostgresSaver.setup()`/`PostgresStore.setup()` both do a
# check-then-insert against a migrations table with no locking of their own — harmless for one
# process, but a genuine race the first time two or more agents run concurrently against a fresh
# database (e.g. `make demo-all`, which starts every pattern's own `get_checkpointer()` call at
# once): both processes see "no migrations yet" and both try to INSERT the same row, and the
# loser crashes with a UniqueViolation instead of just seeing the already-applied migration.
# Different keys for the two locks so a concurrent checkpointer setup and store setup can't
# block each other unnecessarily.
_CHECKPOINTER_SETUP_LOCK_KEY = 727_401
_STORE_SETUP_LOCK_KEY = 727_402


class _SetsUp(Protocol):
    def setup(self) -> None: ...


def _setup_with_advisory_lock[T: _SetsUp](resource: T, uri: str, lock_key: int) -> T:
    """Run `resource.setup()` inside a Postgres advisory lock, then return `resource`.

    Shared by `get_checkpointer`/`get_store`: both do the identical "idempotent `.setup()`,
    guarded by an advisory lock so concurrent first-time callers serialize on the migration
    instead of racing it" dance (see `_CHECKPOINTER_SETUP_LOCK_KEY`'s comment above for why the
    lock exists at all) — they differ only in which resource and which lock key.
    """
    with psycopg.connect(uri, autocommit=True) as lock_conn:
        lock_conn.execute("SELECT pg_advisory_lock(%s)", (lock_key,))
        try:
            resource.setup()
        finally:
            lock_conn.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
    return resource


@contextmanager
def get_checkpointer(postgres_uri: str | None = None) -> Iterator[PostgresSaver]:
    """Yield a `PostgresSaver`, running `.setup()` idempotently on first use.

    `.setup()` itself is wrapped in a Postgres advisory lock (see `_CHECKPOINTER_SETUP_LOCK_KEY`'s
    comment above) so concurrent first-time callers serialize on the migration instead of racing
    it.

    Args:
        postgres_uri: Override the connection string; defaults to `Settings.postgres_uri`.

    Yields:
        A `PostgresSaver` ready to pass as `checkpointer=` to `create_agent` / `StateGraph.compile`.
    """
    uri = postgres_uri or get_settings().postgres_uri
    with PostgresSaver.from_conn_string(uri) as checkpointer:
        yield _setup_with_advisory_lock(checkpointer, uri, _CHECKPOINTER_SETUP_LOCK_KEY)


@contextmanager
def get_store(postgres_uri: str | None = None) -> Iterator[PostgresStore]:
    """Yield a `PostgresStore` for long-term, cross-thread memory.

    `.setup()` itself is wrapped in a Postgres advisory lock, same convention and reasoning as
    `get_checkpointer` — see `_STORE_SETUP_LOCK_KEY`'s comment above.

    Args:
        postgres_uri: Override the connection string; defaults to `Settings.postgres_uri`.

    Yields:
        A `PostgresStore` ready to pass as `store=` to `create_agent` / `StateGraph.compile`.
    """
    uri = postgres_uri or get_settings().postgres_uri
    with PostgresStore.from_conn_string(uri) as store:
        yield _setup_with_advisory_lock(store, uri, _STORE_SETUP_LOCK_KEY)
