# src/llm/cost_tracker.py

"""
LLM Cost Tracker.

Responsible for aggregating token usage and calculating the estimated
cost of LLM operations based on provider pricing.
"""

from typing import Dict, Any, Optional
from config.settings import LLMProvider

from datetime import datetime, timezone

# NOTE: Pricing data (Section 4.4) is hardcoded here. In a production system,
# this should ideally be fetched from a configuration service or an updated source,
# but hardcoding it simplifies the current implementation based on the spec.
# Prices are in USD per 1 Million (M) tokens.
LLM_PRICING_USD_PER_M = {
    # Groq (Llama 3.3 70B - assuming latest pricing, example rates from spec)
    (LLMProvider.GROQ, "input"): 0.59,
    (LLMProvider.GROQ, "output"): 0.59,
    # OpenAI (gpt-4o-2024-11-20 - assumed production model)
    (LLMProvider.OPENAI, "input"): 5.00,  # Example cost
    (LLMProvider.OPENAI, "output"): 15.00,  # Example cost
    # Anthropic (claude-sonnet-4 - assumed production model)
    (LLMProvider.ANTHROPIC, "input"): 3.00,
    (LLMProvider.ANTHROPIC, "output"): 15.00,  # Example cost
    # Anthropic Caching (Section 4.3)
    (LLMProvider.ANTHROPIC, "cache_read"): 0.30,
    (LLMProvider.ANTHROPIC, "cache_write"): 3.75,
}


class CostTracker:
    """
    Tracks token usage and calculates estimated cost for LLM interactions.
    """

    def __init__(self):
        """Initialize all usage metrics to zero."""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.total_cost_usd = 0.0
        self.llm_calls: list[Dict[str, Any]] = []

    def _calculate_cost(
        self,
        provider: LLMProvider,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
    ) -> float:
        """
        Calculate the estimated cost for a single LLM interaction.
        """
        # All prices are per 1,000,000 tokens. Convert token counts to M tokens.
        M_TOKENS = 1_000_000

        # Standard token costs
        input_cost = (prompt_tokens / M_TOKENS) * LLM_PRICING_USD_PER_M.get(
            (provider, "input"), 0
        )
        output_cost = (completion_tokens / M_TOKENS) * LLM_PRICING_USD_PER_M.get(
            (provider, "output"), 0
        )

        # Anthropic Caching costs (Section 4.3)
        cache_read_cost = (
            (cache_read_tokens / M_TOKENS)
            * LLM_PRICING_USD_PER_M.get((provider, "cache_read"), 0)
            if provider == LLMProvider.ANTHROPIC
            else 0
        )
        cache_write_cost = (
            (cache_write_tokens / M_TOKENS)
            * LLM_PRICING_USD_PER_M.get((provider, "cache_write"), 0)
            if provider == LLMProvider.ANTHROPIC
            else 0
        )

        return input_cost + output_cost + cache_read_cost + cache_write_cost

    def record_usage(
        self,
        provider: LLMProvider,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        # Anthropic specific
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        step_name: Optional[str] = None,
    ):
        """
        Records the token usage, calculates cost, and updates totals.
        """
        cost = self._calculate_cost(
            provider,
            prompt_tokens,
            completion_tokens,
            cache_read_tokens,
            cache_write_tokens,
        )

        # Update totals
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cache_read_tokens += cache_read_tokens
        self.cache_write_tokens += cache_write_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.total_cost_usd += cost

        # Store detailed call log for the audit trail (Section 8.2)
        self.llm_calls.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": step_name,
                "provider": provider.value,
                "model": model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
                "cost_usd": cost,
                "latency_ms": latency_ms,
            }
        )

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns the aggregated token usage and cost as a dictionary
        suitable for the ProcessingMetadata model.
        """
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "estimated_cost_usd": self.total_cost_usd,
            # Note: llm_provider/model/duration must be filled in the main workflow
            # which has the final execution context.
        }