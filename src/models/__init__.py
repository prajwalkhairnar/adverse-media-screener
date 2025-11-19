# src/models/__init__.py

"""
Data models for the adverse media screening system.

Defines Pydantic models for inputs, intermediate states, and final output.
"""

from .inputs import ScreeningQuery, LLMProvider  # Defined in inputs.py
from .outputs import (
    ArticleMetadata,
    PersonEntity,
    MatchAssessment,
    SentimentAssessment,
    ProcessingMetadata,
    ScreeningResult,
)  # Defined in outputs.py

# Public API for the models package
__all__ = [
    "ScreeningQuery",
    "LLMProvider",
    "ArticleMetadata",
    "PersonEntity",
    "MatchAssessment",
    "SentimentAssessment",
    "ProcessingMetadata",
    "ScreeningResult",
]