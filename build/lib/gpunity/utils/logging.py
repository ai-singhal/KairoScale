"""Logging configuration for GPUnity."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, verbose: bool = False) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Logger name (typically module path like 'gpunity.profiler').
        verbose: If True, set level to DEBUG; otherwise INFO.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger
