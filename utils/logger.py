"""
utils/logger.py - Logging configuration for JazzBot
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger"""
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Global logger instance
logger = setup_logger(__name__, "jazzbot.log")
