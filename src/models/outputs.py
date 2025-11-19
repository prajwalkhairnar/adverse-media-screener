# src/models/outputs.py

"""
Pydantic models for intermediate and final system outputs.

These models define the structured, traceable output required for
regulatory compliance and system integration.
"""

from typing import Optional, Literal
from datetime import date, datetime
from pydantic import BaseModel, Field

# =============================================================================
# Intermediate Models (Section 3.2)
# =============================================================================


class ArticleMetadata(BaseModel):
    """Extracted article information."""

    url: str = Field(description="Original news article URL.")
    title: str = Field(description="Title of the article.")
    source: str = Field(description='Source publication, e.g., "New York Times".')
    publish_date: Optional[date] = Field(
        default=None, description="Date of publication (YYYY-MM-DD)."
    )
    language: str = Field(description='ISO 639-1 code (e.g., "en", "es").')
    text_content: str = Field(description="Clean, extracted text content.")
    word_count: int = Field(description="Total word count of the extracted content.")


class PersonEntity(BaseModel):
    """Person mentioned in article, extracted by LLM."""

    full_name: str = Field(description="Exact name as found in the article.")
    age: Optional[int] = Field(default=None, description="Specific age if mentioned.")
    approximate_age_range: Optional[str] = Field(
        default=None, description='E.g., "40s", "mid-50s".'
    )
    occupation: Optional[str] = Field(
        default=None, description="Job title, role, or description."
    )
    location: Optional[str] = Field(
        default=None, description="City, country, or region if mentioned."
    )
    context_snippet: str = Field(
        description="1-2 sentences surrounding the mention for context."
    )
    # NOTE: mention_positions from spec (list[int]) is omitted here for simplicity
    # but could be added for advanced auditing if required.
    other_details: Optional[list[str]] = Field(
        default=None, description="Any other identifying details (education, company, etc.)"
    )


class MatchAssessment(BaseModel):
    """Result of matching query person to an article entity (Section 3.2)."""

    is_match: bool = Field(
        description="True if confident match OR uncertain potential match."
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Confidence level of the match decision."
    )
    match_probability: float = Field(
        description="0.0 to 1.0 probability score (for quantitative comparison)."
    )
    reasoning_steps: list[str] = Field(
        description="Chain of thought: step-by-step reasoning for the decision."
    )
    supporting_evidence: list[str] = Field(
        description="List of facts supporting the match (e.g., matching DOB)."
    )
    contradicting_evidence: list[str] = Field(
        description="List of facts contradicting the match (e.g., different age)."
    )
    missing_information: list[str] = Field(
        description="What information is missing that prevents a HIGH confidence match."
    )
    matched_entity: Optional[PersonEntity] = Field(
        default=None, description="The specific article entity that was matched against."
    )


class SentimentAssessment(BaseModel):
    """Sentiment and adverse media analysis result (Section 3.2)."""

    classification: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    is_adverse_media: bool = Field(
        description="True if the article contains adverse media indicators."
    )
    severity: Optional[Literal["HIGH", "MEDIUM", "LOW"]] = Field(
        default=None, description="Severity of the adverse media (if present)."
    )
    adverse_indicators: list[str] = Field(
        description='Specific issues found, e.g., ["fraud", "lawsuit"].'
    )
    evidence_snippets: list[str] = Field(
        description="Quotes from the article supporting the adverse finding."
    )
    reasoning: str = Field(
        description="Explanation of the sentiment and adverse media classification."
    )

# =============================================================================
# Output Models (Section 3.3)
# =============================================================================


class ProcessingMetadata(BaseModel):
    """Execution and cost metadata (Section 3.3)."""

    timestamp: datetime = Field(description="UTC timestamp of when processing finished.")
    total_duration_ms: float = Field(
        description="Total duration of the screening process in milliseconds."
    )

    # LLM usage
    llm_provider: str = Field(description="The final LLM provider used.")
    llm_model: str = Field(description="The specific LLM model used.")
    total_tokens: int = Field(description="Total tokens (prompt + completion).")
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated_cost_usd: float = Field(
        description="Estimated cost of the entire run in USD."
    )

    # Processing steps
    steps_completed: list[str] = Field(description="List of all nodes/steps executed.")
    errors_encountered: list[str] = Field(
        description="List of errors/exceptions encountered."
    )
    warnings: list[str] = Field(description="List of non-critical warnings.")


class ScreeningResult(BaseModel):
    """Complete, final screening result (Section 3.3)."""

    # NOTE: ScreeningQuery (the input model) will be imported and used here,
    # but for simplicity, we use dict structure to avoid circular import dependency with inputs.py
    # and just assume a dict or similar structure is passed here from the state.
    query: dict = Field(description="The original ScreeningQuery used for the run.")
    
    decision: Literal["MATCH", "NO_MATCH", "UNCERTAIN"] = Field(
        description="Final decision based on match assessment."
    )
    
    match_assessment: MatchAssessment
    sentiment_assessment: Optional[SentimentAssessment] = Field(
        default=None, description="Only present if match_assessment determined a match."
    )
    
    article_metadata: ArticleMetadata
    entities_found: list[PersonEntity] = Field(
        description="All people entities extracted from the article."
    )
    
    processing_metadata: ProcessingMetadata
    report: str = Field(
        description="The final human-readable formatted report (Section 5.4)."
    )