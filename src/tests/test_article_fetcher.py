import pytest
from datetime import date
from unittest.mock import MagicMock

# Import the class being tested and its expected output model
from src.utils.article_fetcher import ArticleFetcher
from src.models.outputs import ArticleMetadata

@pytest.fixture
def mock_fetcher_data(mocker):
    """Mocks the external trafilatura library calls."""
    # Mock trafilatura.fetch_url to return mock HTML
    mocker.patch(
        "trafilatura.fetch_url", 
        return_value="<html><body>Article content here.</body></html>"
    )
    # Mock trafilatura.extract to return mock text and metadata
    mock_metadata = {
        'title': 'Test Article Title',
        'source': 'Test Source',
        'date': '2024-01-01T12:00:00Z',
    }
    mocker.patch(
        "trafilatura.extract", 
        return_value=("Article text content here. It has 7 words.", mock_metadata)
    )
    # Mock langdetect.detect to ensure consistent language result
    mocker.patch(
        "src.utils.article_fetcher.detect", 
        return_value="en"
    )

@pytest.mark.asyncio
async def test_article_fetcher_success(mock_fetcher_data):
    """Tests successful fetching and parsing into the ArticleMetadata model."""
    fetcher = ArticleFetcher()
    test_url = "http://example.com/test-article"
    
    metadata = fetcher.fetch_article_data(test_url)
    
    assert isinstance(metadata, ArticleMetadata)
    assert metadata.url == test_url
    assert metadata.title == "Test Article Title"
    assert metadata.source == "Test Source"
    assert metadata.publish_date == date(2024, 1, 1)
    assert metadata.language == "en"
    assert "Article text content here" in metadata.text_content
    assert metadata.word_count == 7

@pytest.mark.asyncio
async def test_article_fetcher_failure_no_text(mock_fetcher_data, mocker):
    """Tests handling a failure where no clean text can be extracted."""
    # Mock trafilatura.extract to return None for content
    mocker.patch(
        "trafilatura.extract", 
        return_value=(None, {})
    )
    
    fetcher = ArticleFetcher()
    
    with pytest.raises(ValueError, match="Failed to extract clean article content"):
        fetcher.fetch_article_data("http://bad-url.com")