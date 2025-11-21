from typing import TypedDict, Optional, Literal, List, Any
from datetime import datetime

from src.models.outputs import (
    ArticleMetadata,
    PersonEntity,
    MatchAssessment,
    SentimentAssessment,
    ScreeningResult,
)
from src.models.inputs import ScreeningQuery

# =============================================================================
# State Definition (Section 7.1)
# =============================================================================

class ScreeningState(TypedDict):
    """
    The state object for the LangGraph workflow.
    It tracks all inputs, intermediate results, and metadata across the screening process.
    """
    # ------------------------------------
    # 1. Input (from ScreeningQuery)
    # ------------------------------------
    query: ScreeningQuery
    
    # ------------------------------------
    # 2. Article processing
    # ------------------------------------
    article_metadata: Optional[ArticleMetadata]
    article_text: Optional[str]
    article_language: Optional[str]
    
    # ------------------------------------
    # 3. Entity extraction
    # ------------------------------------
    entities: List[PersonEntity]
    extraction_complete: bool
    
    # ------------------------------------
    # 4. Matching
    # ------------------------------------
    match_assessment: Optional[MatchAssessment]
    match_decision: Optional[Literal["MATCH", "NO_MATCH", "UNCERTAIN"]]
    
    # ------------------------------------
    # 5. Sentiment (conditional)
    # ------------------------------------
    sentiment_assessment: Optional[SentimentAssessment]
    
    # ------------------------------------
    # 6. Enrichment (Part 2 - future)
    # ------------------------------------
    enrichment_needed: bool # Part 2 - Design Only
    enrichment_data: Optional[dict] # Part 2 - Design Only
    
    # ------------------------------------
    # 7. Report
    # ------------------------------------
    final_screening_result: Optional[ScreeningResult]
    report_complete: Optional[bool]
    
    # ------------------------------------
    # 8. Metadata / Observability
    # ------------------------------------
    errors: List[str]
    warnings: List[str]
    start_time: datetime
    total_duration_ms: Optional[float]
    steps_completed: List[str]

    llm_calls: List[dict] # Detailed log of all LLM usage (from CostTracker)
    
    # Runtime context (used internally for nodes)
    llm_provider: Optional[str]
    llm_model: Optional[str]
    