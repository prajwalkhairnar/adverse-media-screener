# src/nodes/report.py

"""
Node responsible for generating the final, human-readable compliance report.
(Section 2.2.5 and 5.4)
"""
from typing import Dict, Any, List
import json

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser
from pydantic import ValidationError
from datetime import datetime, timezone

from src.graph.state import ScreeningState
from src.nodes.base import BaseNode
from src.utils.logger import get_logger
from config.settings import LLMProvider
from src.models.outputs import ScreeningResult # Final output model
from src.chains.report_generation import create_report_generation_chain


logger = get_logger("ReportNode")


class ReportGenerationNode(BaseNode):
    """
    Node that takes the complete state (assessments, entities, metadata)
    and uses an LLM to generate the final human-readable report.
    """
    
    def __init__(self, llm: BaseLanguageModel, settings: Any, cost_tracker: Any):
        super().__init__(llm, settings, cost_tracker)
        
        # The chain generates a string (the report text)
        self.chain = create_report_generation_chain(llm)

    def run(self, state: ScreeningState, llm_provider: LLMProvider) -> Dict[str, Any]:
        """
        Executes the report generation process and updates the state with the final result.
        """
        logger.info("Running Report Generation Node...")
        


        # 1. Check if the core decision (MATCH, NO_MATCH, UNCERTAIN) has been made
        final_decision = state.get("match_decision")

        # The decision is required to generate a coherent report
        if not final_decision:
            error_msg = "Cannot generate report: Final decision is missing from state."
            logger.error(error_msg)
            return {
                "errors": state["errors"] + [error_msg],
                "report_complete": False,
            }

        # 2. Prepare the input variables for the report prompt
        report_data = {
            "query_info": state["query"].model_dump(),
            "article_metadata": state["article_metadata"].model_dump(),
            "entities": [e.model_dump() for e in state["entities"]],
            "match_assessment": state["match_assessment"].model_dump(),
            "sentiment_assessment": state["sentiment_assessment"].model_dump() 
                                    if state["sentiment_assessment"] else "None (No match detected)",
            "final_decision": final_decision,
            "all_errors": state["errors"],
            "all_warnings": state["warnings"],
        }
        
        # The prompt only expects the 'results_json' variable.
        prompt_vars = {
            "results_json": json.dumps(report_data, indent=2), # Dump as clean JSON string
        }

        # 3. Execute the chain
        try:
            report_text = self._invoke_chain_with_tracking(
                self.chain, 
                prompt_vars, 
                step_name="report_generation",
                llm_provider=llm_provider,
                llm_model=state["llm_model"],
            )

            # 4. Construct the final ScreeningResult model (Section 3.3)
            # 4a. Compile the complete processing_metadata dictionary
            cost_metadata = self.cost_tracker.get_metadata()
            
            processing_metadata = {
                # Workflow Status & Audit
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_duration_ms": state.get("total_duration_ms", 0), # Assumed to be added by workflow
                "steps_completed": state.get("steps_completed", []) + ["report_generation"],
                "errors_encountered": state.get("errors", []),
                "warnings": state.get("warnings", []),
                "llm_calls": state.get("llm_calls", []), # Audit trail from all nodes
                
                # Global/Config Info
                "llm_provider": llm_provider.value,
                "llm_model": state.get("llm_model", "Unknown"),
                
                # Token/Cost data (merging in from the CostTracker)
                **cost_metadata,
            }

            final_result = ScreeningResult(
                query=state["query"].model_dump(),
                decision=final_decision,
                match_assessment=state["match_assessment"],
                sentiment_assessment=state["sentiment_assessment"],
                article_metadata=state["article_metadata"],
                entities_found=state["entities"],
                processing_metadata=processing_metadata,
                report=report_text,
            )
            
            
            logger.info("Final Compliance Report generated successfully.")

            return {
                "final_screening_result": final_result,
                "report_complete": True,
                "steps_completed": state.get("steps_completed", []) + ["report_generation"],
            }
            
        except Exception as e:
            error_msg = f"Report generation failed: {e.__class__.__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            return {
                "errors": state["errors"] + [error_msg],
                "report_complete": False,
            }