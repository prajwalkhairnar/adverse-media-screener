# src/models/inputs.py

"""
Pydantic models for system inputs.

Defines the structure of the data submitted by the user/analyst
for adverse media screening.
"""

from typing import Optional
from pydantic import BaseModel, Field


from enum import Enum
class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ScreeningQuery(BaseModel):
    """Input for adverse media screening (Section 3.1)."""

    # Core required inputs
    name: str = Field(description="Full name of the person being screened.")
    dob: str = Field(description="Date of birth in YYYY-MM-DD format.")
    url: str = Field(description="News article URL to be screened.")

    # Optional overrides
    provider: Optional[LLMProvider] = Field(
        default=None, description="Optional override for the default LLM provider."
    )
    model: Optional[str] = Field(
        default=None, description="Optional override for the default LLM model name."
    )
    enable_enrichment: bool = Field(
        default=False,
        description="Flag to enable automated web enrichment (Part 2 feature)."
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        extra = "forbid"