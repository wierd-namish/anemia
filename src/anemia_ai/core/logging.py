"""
Structured logging configuration for Anemia AI.
"""

import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str = "anemia_ai",
    level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Configures and returns a structured logger instance."""
    if level is None:
        level = os.getenv("ANEMIA_LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, level, logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if already attached
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)

        if log_format is None:
            log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

        formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


logger = setup_logger()
