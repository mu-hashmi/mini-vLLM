"""Logging utilities using loguru."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO", format_string: str | None = None) -> None:
    """Configure loguru logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string. If None, uses default.
    """
    # Remove default handler
    logger.remove()

    # Default format if not provided
    if format_string is None:
        format_string = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"

    # Add console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True,
    )
