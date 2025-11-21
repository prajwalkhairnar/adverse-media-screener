# src/utils/article_fetcher.py

"""
Article Fetcher Utility.

Fetches a news article URL, cleans the HTML content, and extracts
text and metadata using trafilatura and langdetect. (Section 2.2.1)
"""

import requests
import json
import trafilatura
from langdetect import detect, LangDetectException
from datetime import date, datetime
from typing import Optional, Any

from src.models.outputs import ArticleMetadata
from src.utils.logger import get_logger

logger = get_logger("ArticleFetcher")


class ArticleFetcher:
    """
    Handles fetching, cleaning, and extracting metadata from a news article URL.
    """

    def __init__(self):
        """Initialize with standard headers for requests."""
        self.headers = {'User-Agent': 'AdverseMediaScreener-Bot/1.0 (Contact: analyst@example.com)'}

    def _get_article_text(self, url: str) -> tuple[Optional[str], Optional[dict]]:
        """Fetch content and extract text/metadata dict using trafilatura."""
        

        try:
            # 1. Fetch content using requests with custom headers
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            downloaded_html = response.text
            
            if not downloaded_html:
                 raise requests.exceptions.RequestException("Received empty content.")
            
            # 2. Extract main text and metadata as a dictionary from the content
            extracted_json_str = trafilatura.extract(
                downloaded_html,
                output_format="json", 
                include_links=False,
                include_comments=False,
            )
            
            if not extracted_json_str:
                logger.warning(f"Trafilatura failed to extract any content from {url}")
                return None, None

            extracted_data = json.loads(extracted_json_str)
            
            if not extracted_data or not extracted_data.get('text'):
                logger.warning(f"Trafilatura output was empty or missing text from {url}")
                return None, None

            text_content = extracted_data.get('text')
            
            return text_content, extracted_data

        except requests.exceptions.HTTPError as e:

            if e.response.status_code == 404:
                logger.error(f"Article not found (404) at {url}")
                raise ValueError("Article URL returned 404 Not Found.") from e
            elif e.response.status_code in [401, 403]:
                logger.warning(f"Access denied (paywall/403) at {url}")
                raise ValueError("Article is likely behind a paywall or access is forbidden.") from e
            else:
                logger.error(f"HTTP Error fetching {url}: {e}")
                raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout fetching {url}: {e}")
            raise ConnectionError("Timeout reached while fetching article.") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"General Request Error fetching {url}: {e}")
            raise ConnectionError(f"Failed to fetch article: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during article fetching: {e}")
            raise RuntimeError(f"Unexpected error during fetching: {e}") from e

    def _detect_language(self, text: str) -> str:
        """Detect language of the text content using langdetect."""
        try:
            return detect(text)
        except LangDetectException:
            logger.warning("Could not detect language. Defaulting to 'en'.")
            return "en"

    def fetch_and_parse(self, url: str) -> ArticleMetadata:
        """
        Main method to fetch, clean, and structure article data into ArticleMetadata.
        """
        logger.info(f"Attempting to fetch and parse article: {url}")
        
        text_content, metadata_dict = self._get_article_text(url)

        if not text_content:
            raise ValueError("Failed to extract clean article content.")

        # 1. Basic properties
        title = metadata_dict.get('title', "Unknown Title") if metadata_dict else "Unknown Title"
        source = metadata_dict.get('source', url.split('/')[2].split('?')[0]) if metadata_dict else url.split('/')[2].split('?')[0]
        
        # 2. Language and word count
        language = self._detect_language(text_content)
        word_count = len(text_content.split())

        # 3. Date handling (Trafilatura returns string, convert to date object)
        publish_date: Optional[date] = None
        date_str = metadata_dict.get('date') if metadata_dict else None
        if date_str:
            try:
                publish_date = datetime.fromisoformat(date_str.split('T')[0]).date()
            except ValueError:
                logger.warning(f"Could not parse date string: {date_str}. Ignoring date.")

        # 4. Final Data Model assembly (Section 3.2)
        return ArticleMetadata(
            url=url,
            title=title,
            source=source,
            publish_date=publish_date,
            language=language,
            text_content=text_content,
            word_count=word_count,
        )