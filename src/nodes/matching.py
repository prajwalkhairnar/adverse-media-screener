from typing import Dict, Any, List, Optional
from datetime import date

from langchain_core.language_models import BaseLanguageModel
from src.chains.name_matching import create_name_matching_chain
from src.models.schemas import NameMatchingOutput
from langchain_core.output_parsers import JsonOutputParser 

from src.graph.state import ScreeningState
from src.nodes.base import BaseNode
from src.utils.logger import get_logger
from src.utils.validators import verify_age_alignment, parse_date
from config.prompts import format_entity_for_prompt
from src.models.outputs import MatchAssessment, PersonEntity
from config.settings import LLMProvider


logger = get_logger("MatchingNode")


class NameMatchingNode(BaseNode):
    """
    Node responsible for comparing the query person to each extracted entity
    using a multi-tier LLM and rule-based strategy. (Section 2.2.3 and 6)
    """

    def __init__(self, llm: BaseLanguageModel, settings: Any, cost_tracker: Any):
        super().__init__(llm, settings, cost_tracker)
        self.output_parser = JsonOutputParser(pydantic_object=NameMatchingOutput)
        
        # Chain is now initialized from the external src/chains package
        self.chain = create_name_matching_chain(llm)
    
    def _get_best_match(
        self, state: ScreeningState, llm_provider: LLMProvider
    ) -> Optional[MatchAssessment]:
        """
        Iterates through all extracted entities and finds the highest confidence match.
        """
        query = state["query"]
        entities = state.get("entities", [])
        article_metadata = state["article_metadata"]
        
        # Parse required dates
        query_dob = parse_date(query.dob)
        article_date = article_metadata.publish_date or date.today()

        if not entities:
            return None

        best_assessment: Optional[MatchAssessment] = None
        
        for entity in entities:
            # 1. Run deterministic rule-based pre-check: Age Verification
            age_check = verify_age_alignment(
                query_dob, article_date, entity.age
            )

            # 2. Prepare input variables for the LLM prompt
            prompt_vars = {
                "query_name": query.name,
                "query_dob": query.dob,
                "article_date": article_date.isoformat(),
                "article_source": article_metadata.source,
                "age_check_result": age_check,
                "entity_xml": format_entity_for_prompt(entity),
                "context_snippet": entity.context_snippet,
                "format_instructions": self.output_parser.get_format_instructions(),
            }

            # 3. Execute the chain for this entity
            try:
                output = self._invoke_chain_with_tracking(
                    self.chain,
                    prompt_vars,
                    step_name=f"match_entity_{entity.full_name[:15]}",
                    llm_provider=llm_provider,
                    llm_model=state["llm_model"],
                )

                parsed_output = NameMatchingOutput.model_validate(output)

                assessment = parsed_output.final_assessment
                assessment.matched_entity = entity # Attach the entity to the assessment

                # 4. Update Best Match
                if (
                    best_assessment is None
                    or assessment.match_probability > best_assessment.match_probability
                ):
                    best_assessment = assessment
            
            except Exception as e:
                logger.error(
                    f"LLM matching failed for entity {entity.full_name}: {e}", 
                    exc_info=True
                )
                state["warnings"].append(f"Matching failed for entity {entity.full_name}. Skipping.")

        return best_assessment

    def run(self, state: ScreeningState, llm_provider: LLMProvider) -> Dict[str, Any]:
        """
        Executes the name matching process and updates the state.
        """

        logger.info("Running Name Matching Node...")
        
        if not state.get("entities"):
            decision: Literal["MATCH", "NO_MATCH", "UNCERTAIN"] = "NO_MATCH"
            logger.info("Match decision: NO_MATCH (No entities found).")
            
            return {
                "match_decision": decision,
                "match_assessment": MatchAssessment(
                    is_match=False,
                    confidence="LOW",
                    match_probability=0.0,
                    reasoning_steps=["No person entities were found in the article to compare against the query person."],
                    supporting_evidence=[],
                    contradicting_evidence=[],
                    missing_information=[],
                    matched_entity=None,
                ),
                "steps_completed": state.get("steps_completed", []) + ["match_person"],
            }

        # Find the best match across all entities
        best_assessment = self._get_best_match(state, llm_provider)
        
        if best_assessment:
            # Determine the final decision based on the match assessment's boolean flag
            if best_assessment.is_match:
                decision = "MATCH" if best_assessment.confidence == "HIGH" else "UNCERTAIN"
            else:
                decision = "NO_MATCH"
            
            logger.info(
                f"Match decision: {decision} (Confidence: {best_assessment.confidence}, Prob: {best_assessment.match_probability:.2f})"
            )
            
            return {
                "match_assessment": best_assessment,
                "match_decision": decision,
                "steps_completed": state.get("steps_completed", []) + ["match_person"],
                "llm_calls": state.get("llm_calls", []) + self.cost_tracker.llm_calls,
            }
        
        # Default case if loop finishes but no best assessment was set
        logger.error("All matching attempts failed due to errors.")
        return {
            "match_decision": "NO_MATCH",
            "errors": state["errors"] + ["Critical: All LLM matching attempts failed."],
            "steps_completed": state.get("steps_completed", []) + ["match_person_failed"],
        }