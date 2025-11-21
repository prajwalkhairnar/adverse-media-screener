# src/llm/factory.py

"""
LLM Client Factory.

Responsible for creating and configuring LangChain LLM clients
(OpenAI, Anthropic, Groq) based on application settings.
"""

from typing import Union, Optional
from functools import lru_cache

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

from config.settings import Settings, LLMProvider


class LLMFactory:
    """
    Manages the creation and configuration of LangChain LLM clients.
    Ensures consistent settings like temperature=0 for compliance.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the factory with the application settings.
        """
        self.settings = settings

    def _get_openai_client(self, model_name: str) -> ChatOpenAI:
        """Create and configure the ChatOpenAI client."""
        return ChatOpenAI(
            model=model_name,
            temperature=self.settings.llm_temperature,
            api_key=self.settings.openai_api_key,

        )

    def _get_anthropic_client(self, model_name: str) -> ChatAnthropic:
        """Create and configure the ChatAnthropic client."""
        client = ChatAnthropic(
            model=model_name,
            temperature=self.settings.llm_temperature,
            api_key=self.settings.anthropic_api_key,
        )

        return client

    def _get_groq_client(self, model_name: str) -> ChatGroq:
        """Create and configure the ChatGroq client."""
        return ChatGroq(
            model_name=model_name,
            temperature=self.settings.llm_temperature,
            api_key=self.settings.groq_api_key,
        )

    @lru_cache(maxsize=3)
    def get_llm(
        self,
        provider: LLMProvider,
        model_name: Optional[str] = None,
    ) -> BaseLanguageModel:
        """
        Get a configured LLM client instance. Caches clients for reuse.

        Args:
            provider: The LLMProvider enum specifying the required provider.
            model_name: Optional model name override. Defaults to settings.

        Returns:
            A configured LangChain BaseLanguageModel instance.

        Raises:
            ValueError: If the provider is invalid or not configured.
        """
        # 1. Determine the final model name
        if not model_name:
            model_name = self.settings.get_model_name(provider)

        # 2. Check if API key is present before initialization
        if not self.settings.validate_provider(provider):
            raise ValueError(f"Provider {provider.value} is not configured (API key missing).")

        # 3. Initialize the correct client
        if provider == LLMProvider.OPENAI:
            return self._get_openai_client(model_name)
        elif provider == LLMProvider.ANTHROPIC:
            return self._get_anthropic_client(model_name)
        elif provider == LLMProvider.GROQ:
            return self._get_groq_client(model_name)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def get_llm_with_fallback(
        self,
        primary_provider: LLMProvider,
    ) -> BaseLanguageModel:
        """
        Attempt to get the primary LLM, falling back to alternatives if configured.
        The actual retry logic is handled by the LangGraph workflow, but this
        method ensures the primary is tried first.
        """
        try:
            # 1. Try primary provider
            return self.get_llm(primary_provider)
        except (ValueError, Exception) as primary_error:
            # 2. Try fallbacks if enabled
            if self.settings.enable_fallback:
                fallback_providers = self.settings.get_fallback_providers(primary_provider)
                for fallback_provider in fallback_providers:
                    try:
                        return self.get_llm(fallback_provider)
                    except Exception:
                        # Continue to next fallback provider on failure
                        continue
                # If all fallbacks failed
                raise ConnectionError(
                    f"All LLM providers failed. Primary error: {primary_error}"
                ) from primary_error
            
            # 3. No fallback or fallback disabled, re-raise primary error
            raise primary_error