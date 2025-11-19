"""
Configuration settings for the adverse media screening system.

Loads configuration from environment variables using Pydantic Settings.
"""

from enum import Enum
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # LLM Provider Configuration
    # -------------------------------------------------------------------------
    default_llm_provider: LLMProvider = Field(
        default=LLMProvider.GROQ,
        description="Default LLM provider to use",
    )

    # API Keys
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")

    # Model names per provider
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model to use",
    )
    openai_model: str = Field(
        default="gpt-4o-2024-11-20",
        description="OpenAI model to use",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model to use",
    )

    # -------------------------------------------------------------------------
    # LLM Behavior
    # -------------------------------------------------------------------------
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0 = deterministic)",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries for failed LLM calls",
    )
    request_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Request timeout in seconds",
    )
    enable_prompt_caching: bool = Field(
        default=True,
        description="Enable prompt caching (Anthropic only)",
    )
    enable_fallback: bool = Field(
        default=False,
        description="Enable provider fallback on failure",
    )

    # -------------------------------------------------------------------------
    # Observability & Logging
    # -------------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARN, ERROR)",
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text)",
    )
    save_logs: bool = Field(
        default=True,
        description="Save logs to file",
    )
    log_file: str = Field(
        default="outputs/screening.log",
        description="Log file path",
    )

    # LangSmith
    langsmith_api_key: Optional[str] = Field(
        default=None,
        description="LangSmith API key for observability",
    )
    langsmith_project: str = Field(
        default="adverse-media-screening",
        description="LangSmith project name",
    )
    langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith tracing",
    )

    # -------------------------------------------------------------------------
    # Article Fetching
    # -------------------------------------------------------------------------
    user_agent: str = Field(
        default="Mozilla/5.0 (compatible; AdverseMediaScreener/1.0)",
        description="User agent for web requests",
    )
    article_fetch_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout for article fetching in seconds",
    )
    max_article_length: int = Field(
        default=50000,
        ge=1000,
        le=200000,
        description="Maximum article length to process (characters)",
    )

    # -------------------------------------------------------------------------
    # Matching Configuration
    # -------------------------------------------------------------------------
    age_tolerance: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Age tolerance for matching (years)",
    )
    min_match_probability: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum probability to consider as potential match",
    )
    high_confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Threshold for high confidence match",
    )
    medium_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for medium confidence match",
    )

    # -------------------------------------------------------------------------
    # Performance & Cost
    # -------------------------------------------------------------------------
    enable_caching: bool = Field(
        default=False,
        description="Enable result caching (not implemented in MVP)",
    )
    max_concurrent_requests: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent LLM requests",
    )
    cost_alert_threshold: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost alert threshold in USD",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format."""
        valid_formats = ["json", "text"]
        v_lower = v.lower()
        if v_lower not in valid_formats:
            raise ValueError(f"Invalid log format: {v}. Must be one of {valid_formats}")
        return v_lower

    def get_available_providers(self) -> list[LLMProvider]:
        """
        Get list of providers with valid API keys.

        Returns:
            List of available LLM providers
        """
        available = []
        if self.groq_api_key:
            available.append(LLMProvider.GROQ)
        if self.openai_api_key:
            available.append(LLMProvider.OPENAI)
        if self.anthropic_api_key:
            available.append(LLMProvider.ANTHROPIC)
        return available

    def get_api_key(self, provider: LLMProvider) -> Optional[str]:
        """
        Get API key for specified provider.

        Args:
            provider: LLM provider

        Returns:
            API key if available, None otherwise
        """
        if provider == LLMProvider.GROQ:
            return self.groq_api_key
        elif provider == LLMProvider.OPENAI:
            return self.openai_api_key
        elif provider == LLMProvider.ANTHROPIC:
            return self.anthropic_api_key
        return None

    def get_model_name(self, provider: LLMProvider) -> str:
        """
        Get model name for specified provider.

        Args:
            provider: LLM provider

        Returns:
            Model name
        """
        if provider == LLMProvider.GROQ:
            return self.groq_model
        elif provider == LLMProvider.OPENAI:
            return self.openai_model
        elif provider == LLMProvider.ANTHROPIC:
            return self.anthropic_model
        return ""

    def validate_provider(self, provider: LLMProvider) -> bool:
        """
        Check if provider is available (has valid API key).

        Args:
            provider: LLM provider to validate

        Returns:
            True if provider is available, False otherwise
        """
        return provider in self.get_available_providers()

    def get_fallback_providers(self, primary: LLMProvider) -> list[LLMProvider]:
        """
        Get fallback providers (all available providers except primary).

        Args:
            primary: Primary provider

        Returns:
            List of fallback providers
        """
        if not self.enable_fallback:
            return []

        available = self.get_available_providers()
        return [p for p in available if p != primary]


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get singleton settings instance.

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings