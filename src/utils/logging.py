"""Provides functions to create and configure loggers."""

import logging
import sys
from typing import Text, Union, Optional
from pathlib import Path

# Constants
DEFAULT_LOG_FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO

def get_console_handler(format_string: str = DEFAULT_LOG_FORMAT) -> logging.StreamHandler:
    """
    Create a console handler with the specified format.

    Args:
        format_string: The format string for log messages

    Returns:
        Configured StreamHandler for console output
    """
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)
    return console_handler

def get_file_handler(
    log_file: Union[str, Path],
    format_string: str = DEFAULT_LOG_FORMAT
) -> logging.FileHandler:
    """
    Create a file handler with the specified format.

    Args:
        log_file: Path to the log file
        format_string: The format string for log messages

    Returns:
        Configured FileHandler for file output
    """
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(format_string)
    file_handler.setFormatter(formatter)
    return file_handler

def get_logger(
    name: Text = __name__,
    log_level: Union[Text, int] = DEFAULT_LOG_LEVEL,
    log_file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file for file output

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate outputs
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(get_console_handler())
    
    if log_file:
        logger.addHandler(get_file_handler(log_file))
    
    logger.propagate = False
    return logger
