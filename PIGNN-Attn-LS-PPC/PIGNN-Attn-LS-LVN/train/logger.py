from __future__ import annotations

import logging
import os
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


_CONSOLE = Console()

# Public console instance for progress bars and other Rich rendering.
console = _CONSOLE


def _level_from_env(default: str = "INFO") -> str:
    return os.getenv("LOG_LEVEL", default).upper()


def configure_logging(
    *,
    name: str = "simplegnn",
    level: str | int | None = None,
    log_file: Optional[str] = None,
    rich_tracebacks: bool = True,
) -> logging.Logger:
    """Configure Rich logging once and return the configured logger.

    - Logs to console via RichHandler.
    - Optionally logs to a file (plain text, no ANSI).

    Idempotent: calling multiple times won't duplicate handlers.
    """

    logger = logging.getLogger(name)

    if level is None:
        level = _level_from_env("INFO")

    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times.
    if getattr(logger, "_rich_configured", False):
        if log_file and not any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_file)
            for h in logger.handlers
        ):
            _add_file_handler(logger, log_file)
        return logger

    logger.propagate = False

    rich_handler = RichHandler(
        console=_CONSOLE,
        rich_tracebacks=rich_tracebacks,
        tracebacks_show_locals=False,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
    )
    rich_handler.setLevel(level)

    logger.addHandler(rich_handler)

    if log_file:
        _add_file_handler(logger, log_file)

    # Silence chatty third-party loggers (MLflow uses Alembic/SQLAlchemy which
    # can spam INFO on startup).
    if os.getenv("QUIET_THIRD_PARTY", "1") not in {"0", "false", "False"}:
        for noisy in (
            "alembic",
            "alembic.runtime.migration",
            "alembic.runtime.plugins",
            "sqlalchemy",
            "mlflow",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    setattr(logger, "_rich_configured", True)
    return logger


def _add_file_handler(logger: logging.Logger, log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logger.level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# Public logger instance to import everywhere.
log = logging.getLogger("simplegnn")
