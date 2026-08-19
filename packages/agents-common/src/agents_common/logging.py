"""Structured logging (structlog) configuration, shared by every agent and provisioning script.

Every agent's `__main__.py` calls `agents_common.observability.configure_mlflow` first thing,
which calls `configure_logging` below as its own first step — so this gets wired up everywhere
with no per-agent changes needed, same "one startup hook" convention `configure_mlflow` already
established for MLflow. `packages/mlflow-server/scripts/*.py` (which don't call `configure_mlflow`)
call `configure_logging` directly instead.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import structlog

from agents_common.config import get_settings

if TYPE_CHECKING:
    from agents_common.config import Settings


def configure_logging(*, settings: Settings | None = None) -> None:
    """Configure structlog (and bridge stdlib `logging` through it) for the current process.

    Call once at process startup — idempotent to call again (e.g. a test importing multiple
    `__main__` modules), since `structlog.configure`/replacing the root logger's handlers is
    itself idempotent.

    Uses the standard structlog/stdlib bridge (see
    https://www.structlog.org/en/stable/standard-library.html) so both `structlog.get_logger()`
    call sites (every `graph.py`/`tools.py`/provisioning script this repo owns) and plain stdlib
    `logging.getLogger()` call sites (e.g. `agents_common.prompts`'s own logger) render through
    the same JSON pipeline, rather than needing two separately-configured logging systems.

    Renders as JSON unconditionally (not console-pretty) — consistent, parseable output whether
    running locally via `make demo`/`make up` or in a container's `docker compose logs`, at the
    cost of prettiness for a human watching a local terminal.

    Logs to stderr, not stdout: every agent's `__main__.py` prints its actual result
    (`print(result[...])`) to stdout as the CLI's real output, meant to be read (or piped) by a
    human or another program — log lines must never land in that stream.

    Args:
        settings: Override settings; defaults to `get_settings()`. `Settings.log_level` (a
            previously-unused field) sets the root/structlog log level.
    """
    settings = settings or get_settings()
    level = logging.getLevelNamesMapping().get(settings.log_level.upper(), logging.INFO)

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)
