from typing import Dict, Any, List
import json

from langchain_core.language_models import BaseLanguageModel
from src.chains.entity_extraction import create_entity_extraction_chain
from src.models.schemas import ExtractionOutput
from langchain_core.output_parsers import JsonOutputParser # Still needed for instructions

from src.graph.state import ScreeningState
from src.nodes.base import BaseNode
from src.utils.logger import get_logger
from config.settings import LLMProvider


logger = get_logger("ExtractionNode")


class EntityExtractionNode(BaseNode):
    """
    Node responsible for extracting all person entities from the article text
    using an LLM with structured output. (Section 2.2.2 and 5.1)
    """
    
    def __init__(self, llm: BaseLanguageModel, settings: Any, cost_tracker: Any):
        super().__init__(llm, settings, cost_tracker)
        
        # Initialize the structured output parser (used only for format instructions)
        self.output_parser = JsonOutputParser(pydantic_object=ExtractionOutput)
        
        # Chain is now initialized from the external src/chains package
        self.chain = create_entity_extraction_chain(llm)

    
    def run(self, state: ScreeningState, llm_provider: LLMProvider) -> Dict[str, Any]:
        """
        Executes the entity extraction process and updates the state.
        """
        logger.info("Running Entity Extraction Node...")
        
        article_metadata = state["article_metadata"]
        article_content = state["article_text"]
        
        if not article_content:
            logger.error("Skipping extraction: Article content is missing.")
            return {
                "errors": state["errors"] + ["Critical: Article content not found for extraction."],
                "extraction_complete": False,
            }

        # Prepare input variables for the prompt
        prompt_vars = {
            "article_url": article_metadata.url,
            "article_title": article_metadata.title,
            "article_source": article_metadata.source,
            "publish_date": article_metadata.publish_date.isoformat() if article_metadata.publish_date else "Unknown",
            "language": article_metadata.language,
            "article_content": article_content,
            "format_instructions": self.output_parser.get_format_instructions(),
        }

        # Execute the chain
        try:
            output = self._invoke_chain_with_tracking(
                self.chain, 
                prompt_vars, 
                step_name="entity_extraction",
                llm_provider=llm_provider,
                llm_model=state["llm_model"],
            )

            parsed_output = ExtractionOutput.model_validate(output)

            # The output is an ExtractionOutput Pydantic model instance
            entities = parsed_output.extracted_entities
            
            
            logger.info(f"Successfully extracted {len(entities)} person entities.")

            return {
                "entities": entities,
                "extraction_complete": True,
                "warnings": state["warnings"] if entities else state["warnings"] + ["No people entities were extracted from the article."],
                "steps_completed": state.get("steps_completed", []) + ["extract_entities"],
            }
        
        except Exception as e:
            error_msg = f"Entity extraction failed: {e.__class__.__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            return {
                "errors": state["errors"] + [error_msg],
                "extraction_complete": False,
            }