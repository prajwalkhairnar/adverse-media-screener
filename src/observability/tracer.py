# src/observability/tracer.py

from src.utils.logger import get_logger
from config.settings import get_settings

logger = get_logger("Tracer")

def setup_tracing_environment():
    """
    Checks for LangSmith configuration and informs the user.
    (Section 4.2: LangSmith Tracing)
    
    NOTE: LangChain/LangGraph automatically use the environment variables
    LANGCHAIN_TRACING_V2=true, LANGCHAIN_PROJECT, and LANGCHAIN_API_KEY
    if they are set in the process environment.
    """
    settings = get_settings()
    
    # In a fully production system, we would ensure these are exported
    # to the OS environment if not already there, but Pydantic Settings
    # gives us the values.
    
    if settings.langsmith_api_key and settings.langsmith_project:
        logger.info(
            f"✅ LangSmith tracing is configured for project: {settings.langsmith_project}"
        )
        # Note: The LangChain runtime must also be set with LANGCHAIN_TRACING_V2=true
    else:
        logger.warning(
            "⚠️ LangSmith tracing is **disabled**. Set LANGCHAIN_API_KEY and "
            "LANGCHAIN_PROJECT in your .env file to enable tracing/observability (Section 4.2)."
        )