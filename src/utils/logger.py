# src/utils/logger.py

"""
Structured Logging Configuration (structlog).

Provides a centralized logger instance configured for structured,
machine-readable, and human-readable output (JSON/Key-Value pairs).
Essential for the audit trail (Section 8.2).
"""

import sys
import logging
from typing import Any

import structlog

from structlog.processors import EventRenamer, dict_tracebacks
from structlog.processors import StackInfoRenderer
from structlog.processors import CallsiteParameterAdder
from structlog.processors import CallsiteParameter
from structlog.stdlib import add_logger_name, add_log_level


from config.settings import get_settings


def key_stripper(keys):
    def processor(logger, method_name, event_dict):
        for key in keys:
            event_dict.pop(key, None)
        return event_dict
    return processor

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def configure_logging():
    """
    Configures structlog to format logs based on environment and settings.
    """
    settings = get_settings()

    # Determine logging style based on environment
    is_development = settings.log_level.upper() in ("DEBUG", "INFO") and sys.stderr.isatty()
    
    # 1. Shared Processors
    shared_processors = [
        # Add the time stamp to the event
        structlog.processors.TimeStamper(fmt="iso"),
        # Add the logger name (e.g., 'ArticleFetcher')
        add_logger_name,
        # Add log level (e.g., 'info', 'error')
        add_log_level,
        # Add exception info for traceback
        structlog.processors.format_exc_info,
        # Add stack info for debugging/audit
        StackInfoRenderer(),
        # Replace Callsites with CallsiteParameterAdder specifying needed params
        CallsiteParameterAdder(parameters=[
            CallsiteParameter.PATHNAME,
            CallsiteParameter.LINENO,
            CallsiteParameter.FUNC_NAME,
        ]),
        # Remove structlog's internal event dict key for cleaner output
        EventRenamer("message"),
        # Key sorting is useful for stable log format
        key_stripper(keys=['_record', '_from_structlog']),
        dict_tracebacks,
    ]

    # 2. Console/Development Processors
    if is_development:
        processors = shared_processors + [
            # Prepare logs for terminal color formatting
            structlog.dev.ConsoleRenderer(colors=True, sort_keys=True)
        ]
    # 3. Production/Audit Processors (JSON)
    else:
        processors = shared_processors + [
            # Render the final log dict as JSON for machine readability (Section 8.2)
            structlog.processors.JSONRenderer(sort_keys=True)
        ]

    # Configure Python's standard logging (needed for LangChain/Requests integration)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.log_level.upper(),
    )

    # Configure structlog
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Returns a configured structlog logger instance.

    Args:
        name: The name of the logger (e.g., the module name).
    """
    # Ensure configuration runs only once
    global _logging_configured
    if not _logging_configured:
        configure_logging()
        _logging_configured = True
        
    return structlog.get_logger(name)

# Initial configuration state
_logging_configured = False

# Run configuration on module import
configure_logging()