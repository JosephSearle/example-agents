"""Unit tests for agents_common.logging — pure logic, no live services."""

from __future__ import annotations

from agents_common.config import Settings
from agents_common.logging import configure_logging
import pytest
import structlog

pytestmark = pytest.mark.unit


def test_configure_logging_marks_structlog_as_configured() -> None:
    configure_logging(settings=Settings(LOG_LEVEL="INFO"))

    assert structlog.is_configured()


def test_configure_logging_is_safe_to_call_twice() -> None:
    configure_logging(settings=Settings(LOG_LEVEL="DEBUG"))
    configure_logging(settings=Settings(LOG_LEVEL="INFO"))

    assert structlog.is_configured()


def test_configure_logging_falls_back_to_info_for_an_unknown_level() -> None:
    # Settings.log_level is a free-text field (no enum validation) — a typo'd LOG_LEVEL in .env
    # shouldn't crash the process at startup.
    configure_logging(settings=Settings(LOG_LEVEL="not-a-real-level"))

    assert structlog.is_configured()
