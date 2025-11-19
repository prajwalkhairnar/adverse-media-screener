# src/utils/__init__.py

"""
Utility package for adverse media screening system.

Contains standalone, reusable helper modules:
- article_fetcher: Logic for fetching and cleaning news article content.
- logger: Centralized logging configuration (structlog).
- validators: Helper functions for data validation (e.g., DOB format, URL checks).
"""

from .article_fetcher import ArticleFetcher  # Defined in article_fetcher.py
from .logger import get_logger  # Defined in logger.py
from .validators import (
    validate_url,
    verify_age_alignment,
)  # Defined in validators.py

# Public API for the utilities package
__all__ = [
    "ArticleFetcher",
    "get_logger",
    "validate_url",
    "verify_age_alignment",
]