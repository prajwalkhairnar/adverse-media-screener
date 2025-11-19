"""
Configuration module for adverse media screening system.

This module contains:
- settings.py: Environment configuration and application settings
- prompts.py: All LLM prompts used in the system
"""

from config.settings import Settings, LLMProvider

__all__ = ["Settings", "LLMProvider"]