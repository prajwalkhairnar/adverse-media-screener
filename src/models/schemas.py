# src/models/schemas.py

"""
Shared Pydantic models for structured input/output of specific workflow steps.
This file is used to break circular dependencies between nodes and chains.
"""

from pydantic import BaseModel, Field
from typing import List


from src.models.outputs import PersonEntity, MatchAssessment, SentimentAssessment


class ExtractionOutput(BaseModel):
    """
    Structured output schema for the Entity Extraction chain. (Section 5.1)
    """
    extracted_entities: List[PersonEntity] = Field(
        description="A list of all unique person entities extracted from the article."
    )



class NameMatchingOutput(BaseModel):
    """
    Structured output schema for the Name Matching chain. (Section 5.2)
    
    This model primarily captures the final assessment derived from the matching process.
    """
    final_assessment: MatchAssessment = Field(
        description="The final match assessment derived from the name matching process."
    )


class SentimentOutput(BaseModel):
    """
    Structured output schema for the Sentiment Analysis chain. (Section 5.3)
    
    This model captures the final sentiment assessment.
    """
    assessment: SentimentAssessment = Field(
        description="The detailed sentiment assessment."
    )