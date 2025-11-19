# src/utils/validators.py

"""
Data validation and utility functions.

Contains reusable functions for input validation and core logic
like age verification for the Name Matching strategy.
"""

from typing import Optional, Literal
from urllib.parse import urlparse
from datetime import date
from dateutil.relativedelta import relativedelta

from src.utils.logger import get_logger # Requires logger.py

logger = get_logger("Validators")


def validate_url(url: str) -> bool:
    """
    Checks if a string is a valid, well-formed HTTP/HTTPS URL.
    """
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except Exception as e:
        logger.warning(f"URL validation failed for {url}: {e}")
        return False


def parse_date(date_str: str, date_format: str = "%Y-%m-%d") -> Optional[date]:
    """
    Safely parses a date string into a date object.
    """
    if not date_str:
        return None
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        logger.warning(f"Failed to parse date string: {date_str} (expected {date_format})")
        return None


def calculate_age(dob: date, date_of_event: date) -> int:
    """
    Calculates the age of a person as of a specific date.
    """
    age = date_of_event.year - dob.year - (
        (date_of_event.month, date_of_event.day) < (dob.month, dob.day)
    )
    return age


def verify_age_alignment(
    query_dob_str: str,
    article_date: date,
    mentioned_age: Optional[int],
    tolerance_years: int = 2,
) -> Literal["MATCH", "MISMATCH", "UNCERTAIN"]:
    """
    Checks if the query DOB aligns with the age mentioned in the article
    based on the article's publication date. (Section 6.2)

    Args:
        query_dob_str: The DOB of the person being screened (YYYY-MM-DD).
        article_date: The date the article was published.
        mentioned_age: The specific age (int) mentioned in the article, if any.
        tolerance_years: The maximum allowed difference in years.

    Returns:
        "MATCH", "MISMATCH", or "UNCERTAIN"
    """
    if not mentioned_age:
        return "UNCERTAIN"

    query_dob = parse_date(query_dob_str)
    if not query_dob:
        logger.warning(f"Invalid query DOB format: {query_dob_str}")
        return "UNCERTAIN"

    # 1. Calculate the query person's age at the time of the article
    calculated_age = calculate_age(query_dob, article_date)
    
    # 2. Check for alignment within tolerance
    age_difference = abs(calculated_age - mentioned_age)

    if age_difference <= tolerance_years:
        return "MATCH"
    else:
        logger.info(
            f"Age mismatch: Calculated age {calculated_age} vs. Mentioned age {mentioned_age}. "
            f"Difference: {age_difference} years (Tolerance: {tolerance_years})."
        )
        return "MISMATCH"