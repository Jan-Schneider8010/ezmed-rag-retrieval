"""Logging setup using rich for readable local output."""

import logging

from rich.logging import RichHandler

from ezmed.settings import settings


def configure_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
