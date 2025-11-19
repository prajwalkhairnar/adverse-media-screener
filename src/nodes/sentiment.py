from typing import Dict, Any, List

from langchain_core.language_models import BaseLanguageModel
# Import chain creation function and required output model
from src.chains.sentiment_analysis import create_sentiment_analysis_chain
from src.models.schemas import SentimentOutput
from langchain_core.output_parsers import JsonOutputParser # Still needed for instructions

from src.graph.state import ScreeningState
from src.nodes.base import BaseNode
from src.utils.logger import get_logger
from src.models.outputs import SentimentAssessment, PersonEntity
from config.settings import LLMProvider


logger = get_logger("SentimentNode")


class SentimentAnalysisNode(BaseNode):
    """
    Node responsible for classifying the article's portrayal of the matched
    individual and identifying adverse media indicators. (Section 2.2.4 and 5.3)
    """

    def __init__(self, llm: BaseLanguageModel, settings: Any, cost_tracker: Any):
        super().__init__(llm, settings, cost_tracker)
        self.output_parser = JsonOutputParser(pydantic_object=SentimentOutput)
        
        # 💡 Chain is now initialized from the external src/chains package
        self.chain = create_sentiment_analysis_chain(llm)

    # NOTE: The _build_chain method is REMOVED.

    def run(self, state: ScreeningState, llm_provider: LLMProvider) -> Dict[str, Any]:
        """
        Executes the sentiment analysis process and updates the state.
        """
        logger.info("Running Sentiment Analysis Node...")

        match_assessment = state["match_assessment"]
        article_text = state["article_text"]
        
        if not match_assessment or not match_assessment.is_match:
            logger.info("Skipping sentiment analysis: No confident match found.")
            return {
                "sentiment_assessment": None,
                "steps_completed": state.get("steps_completed", []) + ["analyze_sentiment_skipped"],
            }

        # The entity that was determined to be the best match
        # We assume the entity is attached in the matching node
        matched_entity: PersonEntity = match_assessment.matched_entity
        person_name = matched_entity.full_name

        # Prepare input variables for the prompt (Section 5.3)
        prompt_vars = {
            "article_text": article_text,
            "person_name": person_name,
            "format_instructions": self.output_parser.get_format_instructions(),
        }

        # Execute the chain
        try:
            output: SentimentOutput = self._invoke_chain_with_tracking(
                self.chain,
                prompt_vars,
                step_name="sentiment_analysis",
                llm_provider=llm_provider,
                llm_model=state["llm_model"],
            )
            
            assessment = output.sentiment_assessment
            
            logger.info(
                f"Sentiment Result: {assessment.classification}, Adverse: {assessment.is_adverse_media}, Severity: {assessment.severity}"
            )
            
            return {
                "sentiment_assessment": assessment,
                "steps_completed": state.get("steps_completed", []) + ["analyze_sentiment"],
                "llm_calls": state.get("llm_calls", []) + self.cost_tracker.get_latest_calls(),
            }

        except Exception as e:
            # ... (Error handling logic is unchanged)
            error_msg = f"Sentiment analysis failed: {e.__class__.__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            return {
                "errors": state["errors"] + [error_msg],
                "sentiment_assessment": SentimentAssessment(
                    classification="NEUTRAL",
                    is_adverse_media=False,
                    severity=None,
                    adverse_indicators=[],
                    evidence_snippets=[],
                    reasoning=f"Sentiment analysis failed due to technical error: {e.__class__.__name__}.",
                ),
                "steps_completed": state.get("steps_completed", []) + ["analyze_sentiment_failed"],
            }