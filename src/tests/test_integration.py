import pytest
import json
from unittest.mock import MagicMock, patch
from datetime import datetime, date
from langchain_core.pydantic_v1 import parse_obj_as

# Import system components
from src.graph.workflow import AdverseMediaWorkflow
from src.llm.cost_tracker import CostTracker
from src.llm.factory import LLMFactory
from src.models.inputs import ScreeningQuery
from src.models.outputs import ArticleMetadata, NameMatchingOutput, SentimentOutput
from config.settings import get_settings, LLMProvider


# --- Fixtures ---

def load_test_case(case_name: str) -> dict:
    """Loads a specific test case from the JSON file (simulated)."""
    # NOTE: In a real test environment, you'd read the file
    with open("tests/test_cases.json", "r") as f:
        return json.load(f)[case_name]

@pytest.fixture
def mock_external_dependencies(mocker):
    """Mocks ArticleFetcher and all LLM chains for E2E test."""
    test_data = load_test_case("simple_query")
    
    # 1. Mock ArticleFetcher.fetch_article_data
    mock_article_metadata = parse_obj_as(
        ArticleMetadata, 
        test_data["expected_fetcher_output"]
    )
    mocker.patch(
        "src.graph.workflow.ArticleFetcher.fetch_article_data", 
        return_value=mock_article_metadata
    )
    
    # 2. Mock LLM chain invokes (the core LLM calls)
    mock_extraction_output = parse_obj_as(NameMatchingOutput, test_data["mock_llm_extraction_output"])
    mock_matching_output = parse_obj_as(NameMatchingOutput, test_data["mock_llm_matching_output"])
    mock_sentiment_output = parse_obj_as(SentimentOutput, test_data["mock_llm_sentiment_output"])
    
    # Mock the LLM chain calls based on step name
    def mock_invoke_chain(chain, input_vars, step_name, llm_provider, llm_model):
        if "extraction" in step_name:
            return mock_extraction_output
        elif "match_entity" in step_name:
            # We must wrap the MatchAssessment in NameMatchingOutput Pydantic model
            return NameMatchingOutput(match_assessment=mock_matching_output.match_assessment)
        elif "sentiment_analysis" in step_name:
            return mock_sentiment_output
        elif "report_generation" in step_name:
            # Report returns a string
            return "This is the final generated report for Jane Doe (MATCH/ADVERSE)."
        
        raise ValueError(f"Unexpected step name in mock: {step_name}")
        
    mocker.patch(
        "src.nodes.base.BaseNode._invoke_chain_with_tracking",
        side_effect=mock_invoke_chain
    )
    
    # 3. Mock the LLMFactory to return a simple mock LLM client
    mock_llm = MagicMock()
    mocker.patch.object(LLMFactory, 'get_llm', return_value=mock_llm)

# --- Tests ---

@pytest.mark.asyncio
async def test_workflow_end_to_end_success(mock_external_dependencies):
    """
    Tests the full LangGraph workflow execution for a clean, adverse match case.
    """
    settings = get_settings()
    llm_factory = LLMFactory(settings)
    cost_tracker = CostTracker()
    
    # Arrange: Load query input
    test_data = load_test_case("simple_query")
    query_dict = test_data["simple_query"]
    query = ScreeningQuery(**query_dict)
    
    # Initialize the workflow
    workflow = AdverseMediaWorkflow(settings, llm_factory, cost_tracker)
    
    # Act: Run the workflow
    final_state = workflow.run_workflow(query)
    
    # Assertions
    
    # 1. Final Decision and Report
    assert final_state["match_decision"] == "MATCH"
    assert "final_screening_result" in final_state
    assert "final_report" in final_state
    
    result = final_state["final_screening_result"]
    assert result.decision == "MATCH"
    assert "Jane Doe" in result.report
    
    # 2. Workflow Path (Fetch -> Extract -> Match -> Sentiment -> Report)
    expected_steps = [
        "fetch_article", 
        "extract_entities", 
        "match_person", 
        "analyze_sentiment", 
        "generate_report"
    ]
    assert all(step in final_state["steps_completed"] for step in expected_steps)
    
    # 3. Match Assessment Check
    assert result.match_assessment.is_match is True
    assert result.match_assessment.confidence == "HIGH"
    
    # 4. Sentiment Assessment Check
    assert result.sentiment_assessment is not None
    assert result.sentiment_assessment.is_adverse_media is True
    assert result.sentiment_assessment.classification == "NEGATIVE"
    
    # 5. Metadata/Cost Check (Cost tracker should have recorded usage)
    assert result.processing_metadata.total_duration_ms > 0
    assert result.processing_metadata.prompt_tokens > 0 # Should have accumulated tokens
    assert len(final_state["llm_calls"]) == 4 # Extraction, Matching, Sentiment, Report