import pytest
import json
from unittest.mock import MagicMock
from langchain_core.pydantic_v1 import parse_obj_as

# Import system components
from src.nodes.extraction import EntityExtractionNode, ExtractionOutput
from src.models.outputs import ArticleMetadata
from src.graph.state import ScreeningState
from src.llm.cost_tracker import CostTracker
from config.settings import LLMProvider, get_settings

# --- Fixtures and Helpers ---

def load_test_case(case_name: str) -> dict:
    """Loads a specific test case from the JSON file (simulated)."""
    # NOTE: In a real test environment, you'd read the file
    with open("tests/test_cases.json", "r") as f:
        return json.load(f)[case_name]

@pytest.fixture
def mock_llm_extraction(mocker):
    """Mocks the LLM's invoke method to return a controlled structured output."""
    test_data = load_test_case("simple_query")
    mock_output_dict = test_data["mock_llm_extraction_output"]
    
    # Create the Pydantic model instance the LLM is expected to return
    mock_pydantic_output = parse_obj_as(ExtractionOutput, mock_output_dict)
    
    # Mock the LLM chain's invoke method
    mock_chain_invoke = mocker.patch(
        "langchain_core.runnables.base.Runnable.invoke", 
        return_value=mock_pydantic_output
    )
    
    # Mock the LLM object itself (to pass to the Node)
    mock_llm = MagicMock()
    mock_llm.invoke = mock_chain_invoke.is_called
    mock_llm.with_structured_output.return_value = mock_llm
    
    return mock_llm, mock_chain_invoke

@pytest.fixture
def initial_state_for_extraction():
    """Provides a valid state ready for the Extraction Node."""
    # Use the ArticleMetadata model for the article data
    metadata = ArticleMetadata(
        url="http://test.com/article",
        title="Test Title",
        source="Test Source",
        publish_date=date(2023, 11, 20),
        language="en",
        text_content="This is the article content to be extracted from.",
        word_count=10
    )
    
    # Create a minimal ScreeningState
    return ScreeningState(
        query=MagicMock(), # Mock out the query
        article_metadata=metadata,
        article_text=metadata.text_content,
        start_time=MagicMock(),
        llm_provider=LLMProvider.OPENAI.value,
        llm_model="gpt-4o-mini",
        errors=[],
        warnings=[],
        steps_completed=[]
    )

# --- Tests ---

def test_extraction_node_success(mock_llm_extraction, initial_state_for_extraction):
    """Tests successful entity extraction and state update."""
    mock_llm, mock_chain_invoke = mock_llm_extraction
    node = EntityExtractionNode(
        llm=mock_llm, 
        settings=get_settings(), 
        cost_tracker=CostTracker()
    )
    
    # Act: Run the node
    updates = node.run(initial_state_for_extraction, LLMProvider.OPENAI)
    
    # Assertions
    assert updates["extraction_complete"] is True
    assert len(updates["entities"]) == 1
    assert updates["entities"][0].full_name == "Jane Doe"
    assert "extract_entities" in updates["steps_completed"]
    
    # Verify the LLM chain was called once
    assert mock_chain_invoke.call_count == 1
    
def test_extraction_node_no_article_content(mock_llm_extraction, initial_state_for_extraction):
    """Tests handling a critical error (missing article content)."""
    mock_llm, _ = mock_llm_extraction
    node = EntityExtractionNode(
        llm=mock_llm, 
        settings=get_settings(), 
        cost_tracker=CostTracker()
    )
    
    # Arrange: Set article_text to None
    initial_state_for_extraction["article_text"] = None
    
    # Act
    updates = node.run(initial_state_for_extraction, LLMProvider.OPENAI)
    
    # Assertions
    assert updates["extraction_complete"] is False
    assert "Critical: Article content not found" in updates["errors"][0]