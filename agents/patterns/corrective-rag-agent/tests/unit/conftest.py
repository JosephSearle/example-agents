"""Unit-test fixtures — everything here avoids real network calls and real Postgres/Milvus."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import InMemorySaver
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def in_memory_checkpointer() -> Iterator[InMemorySaver]:
    """An in-process checkpointer, standing in for PostgresSaver in fast unit tests."""
    yield InMemorySaver()
