"""
Logging Utilities - Configures rich logging for the simulation.
"""

import logging
import sys
from typing import Optional

try:
    from rich.logging import RichHandler
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
):
    """
    Set up logging configuration for the simulation.
    Uses rich for console output if available.
    
    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to a log file.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = []

    # Console handler
    if HAS_RICH:
        console = Console(width=120)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=False,
            rich_tracebacks=True,
        )
        console_handler.setLevel(log_level)
        handlers.append(console_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    # File handler
    if log_file:
        from pathlib import Path

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,
    )

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={level}, file={log_file}")
