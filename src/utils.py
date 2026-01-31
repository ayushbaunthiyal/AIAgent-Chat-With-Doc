"""Utility functions for the RAG Chat Assistant."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

from src.config import settings


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def validate_file_extension(file_path: str, allowed_extensions: list[str]) -> bool:
    """Validate if file has an allowed extension."""
    return Path(file_path).suffix.lower() in [ext.lower() for ext in allowed_extensions]


def sanitize_text(text: str) -> str:
    """Sanitize text input to prevent prompt injection."""
    # Basic sanitization - can be enhanced
    text = text.strip()
    # Remove potential prompt injection patterns
    dangerous_patterns = ["<|", "|>", "<<", ">>"]
    for pattern in dangerous_patterns:
        text = text.replace(pattern, "")
    return text


def format_metadata(metadata: Dict[str, Any]) -> str:
    """Format metadata dictionary for display."""
    return ", ".join(f"{k}: {v}" for k, v in metadata.items() if v is not None)
