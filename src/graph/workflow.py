from typing import Literal, Any, Dict, List
from functools import partial
from datetime import datetime # Added missing import

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig

from src.graph.state import ScreeningState
from src.nodes.extraction import EntityExtractionNode
from src.nodes.matching import NameMatchingNode
from src.nodes.sentiment import SentimentAnalysisNode
from src.nodes.report import ReportGenerationNode
from src.utils.logger import get_logger
from src.utils.article_fetcher import ArticleFetcher
from src.models.inputs import ScreeningQuery
from src.llm.cost_tracker import CostTracker
from src.llm.factory import LLMFactory

logger = get_logger("Workflow")


class AdverseMediaWorkflow:
    """
    The orchestrator for the adverse media screening system, implemented using LangGraph.
    (Section 2.1: High-Level Architecture, Section 7: Workflow State Machine)
    """

    def __init__(self, settings: Any, llm_factory: LLMFactory, cost_tracker: CostTracker):
        """
        Initializes the workflow with dependencies and builds the graph.
        """
        self.settings = settings
        self.llm_factory = llm_factory
        self.cost_tracker = cost_tracker
        self.graph = self._build_graph()

    def _get_llm(self, provider: str) -> BaseLanguageModel:
        """Helper to get the configured LLM client."""
        return self.llm_factory.get_llm(provider)

    # =========================================================================
    # Nodes (Section 7.2)
    # =========================================================================

    def fetch_article_node(self, state: ScreeningState) -> Dict[str, Any]:
        """Node 1: Fetches article content and metadata."""
        logger.info("Executing node: fetch_article_node")
        query: ScreeningQuery = state["query"]
        article_fetcher = ArticleFetcher()
        try:
            # Note: ArticleFetcher includes its own retry logic.
            metadata = article_fetcher.fetch_and_parse(query.url)
            
            return {
                "article_metadata": metadata,
                "article_text": metadata.text_content,
                "article_language": metadata.language,
                "steps_completed": ["fetch_article"],
            }
        except Exception as e:
            error_msg = f"Article fetching failed for URL {query.url}: {e}"
            logger.error(error_msg, exc_info=True)
            # Critical error: cannot proceed without article text
            return {
                "errors": state["errors"] + [error_msg],
                "match_decision": "NO_MATCH", # Force end on failure
                "steps_completed": ["fetch_article_failed"],
            }

    def _get_llm_chain_node(self, NodeClass: Any) -> partial:
        """Creates a partial function for LLM-based nodes."""
        
        # Determine the primary LLM provider/model from the current state/settings
        # FIX: Changed the line below to directly use the enum attribute, 
        # removing the unnecessary and non-existent 'get_provider_enum' method call.
        primary_provider = self.settings.default_llm_provider
        
        llm = self.llm_factory.get_llm(primary_provider)

        # Initialize the node with LLM, settings, and cost tracker
        node_instance = NodeClass(
            llm=llm, 
            settings=self.settings, 
            cost_tracker=self.cost_tracker
        )
        
        # Return a partial function that accepts only the 'state' argument
        # This is the LangGraph standard signature for nodes
        return partial(node_instance.run, llm_provider=primary_provider)

    # --- FIX: Changed assignments to instance methods to ensure 'self' is bound ---

    def extract_entities_node(self) -> partial:
        return self._get_llm_chain_node(NodeClass=EntityExtractionNode)
    
    def match_person_node(self) -> partial:
        return self._get_llm_chain_node(NodeClass=NameMatchingNode)
    
    def analyze_sentiment_node(self) -> partial:
        return self._get_llm_chain_node(NodeClass=SentimentAnalysisNode)

    def generate_report_node(self) -> partial:
        return self._get_llm_chain_node(NodeClass=ReportGenerationNode)


    # =========================================================================
    # Conditional Edges (Section 7.3)
    # =========================================================================

    def route_match_decision(self, state: ScreeningState) -> Literal["sentiment", "report"]:
        """
        Determines the next step based on the Name Matcher's output.
        - If MATCH/UNCERTAIN, proceed to sentiment analysis.
        - If NO_MATCH (or critical article fetch failure), skip sentiment and go to report.
        """
        decision = state.get("match_decision")
        
        if decision in ["MATCH", "UNCERTAIN"]:
            logger.info("Match or Uncertain. Proceeding to sentiment analysis.")
            return "sentiment"
        elif decision == "NO_MATCH":
            logger.info("No Match found. Skipping sentiment analysis and proceeding to report.")
            return "report"
        else:
             # Should be caught by error handling in the node, but defensive check
            logger.error(f"Invalid match decision in state: {decision}. Proceeding to report.")
            return "report"

    # NOTE: The Part 2 feature 'enrich_data' node and routing logic 
    # (match_person -> enrich_data -> match_person) are omitted for the MVP 
    # but the state has reserved fields for it.

    # =========================================================================
    # Graph Builder
    # =========================================================================

    def _build_graph(self) -> StateGraph:
        """
        Builds the LangGraph StateGraph (Section 7.3).
        """
        workflow = StateGraph(ScreeningState)

        # 1. Add Nodes (Section 7.2)
        workflow.add_node("fetch_article", self.fetch_article_node)
        workflow.add_node("extract_entities", self.extract_entities_node())
        workflow.add_node("match_person", self.match_person_node())
        workflow.add_node("analyze_sentiment", self.analyze_sentiment_node())
        workflow.add_node("generate_report", self.generate_report_node())

        # 2. Set Edges (Flow: START -> Fetch -> Extract -> Match)
        workflow.set_entry_point("fetch_article")
        workflow.add_edge("fetch_article", "extract_entities")
        workflow.add_edge("extract_entities", "match_person")

        # 3. Add Conditional Edge from Matching Node
        workflow.add_conditional_edges(
            "match_person",
            self.route_match_decision, # Route logic (Section 7.3)
            {
                "sentiment": "analyze_sentiment",
                "report": "generate_report",
            },
        )

        # 4. Final Edges
        workflow.add_edge("analyze_sentiment", "generate_report")
        workflow.add_edge("generate_report", END)

        # 5. Compile the graph
        return workflow.compile()

    # =========================================================================
    # Public Runner
    # =========================================================================

    def run_workflow(self, query: ScreeningQuery) -> ScreeningState:
        """
        Runs the compiled workflow from start to finish.
        
        Args:
            query: The validated input ScreeningQuery.

        Returns:
            The final state dictionary.
        """
        logger.info(f"Starting workflow for: {query.name}, URL: {query.url}")
        
        # Initialize the state (Section 7.1)
        initial_state: ScreeningState = {
            "query": query,
            "article_metadata": None,
            "article_text": None,
            "article_language": None,
            "entities": [],
            "extraction_complete": False,
            "match_assessment": None,
            "match_decision": None,
            "sentiment_assessment": None,
            "enrichment_needed": False,
            "enrichment_data": None,
            "final_report": None,
            "errors": [],
            "warnings": [],
            "start_time": datetime.utcnow(),
            "llm_calls": [],
            # These are set in main.py but are useful for nodes
            "llm_provider": query.provider.value if query.provider else self.settings.default_llm_provider.value,
            "llm_model": query.model or self.settings.get_model_name(query.provider or self.settings.default_llm_provider),
        }

        # The config is used for tracing with LangSmith (Section 4.2)
        config: RunnableConfig = {
            "configurable": {
                "session_id": f"screener-{initial_state['start_time'].isoformat().replace(':', '-')}"
            },
            "tags": [
                f"provider:{initial_state['llm_provider']}",
                f"model:{initial_state['llm_model']}",
            ],
            "metadata": {
                "user_name": query.name,
                "article_url": query.url,
            }
        }
        
        # Run the graph
        final_state: ScreeningState = self.graph.invoke(
            initial_state, 
            config=config,
            # Max steps ensures we don't run into an infinite loop (e.g., in a future enrichment loop)
            # Here it's 5 nodes maximum, set higher for safety.
            recursion_limit=10 
        )
        
        return final_state