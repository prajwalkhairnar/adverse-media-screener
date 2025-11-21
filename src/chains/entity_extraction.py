from typing import Type
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel


from config.prompts import ENTITY_EXTRACTION_SYSTEM_PROMPT, ENTITY_EXTRACTION_USER_PROMPT
from src.nodes.base import BaseNode
from src.models.schemas import ExtractionOutput

def create_entity_extraction_chain(llm: BaseLanguageModel) -> Runnable:
    """
    Creates the LangChain Runnable for Entity Extraction (Section 5.1).

    Args:
        llm: The configured LLM (e.g., ChatOpenAI, ChatAnthropic).

    Returns:
        A LangChain Runnable that takes prompt variables and returns an ExtractionOutput model.
    """
    output_parser = JsonOutputParser(pydantic_object=ExtractionOutput)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ENTITY_EXTRACTION_SYSTEM_PROMPT),
            ("human", ENTITY_EXTRACTION_USER_PROMPT),
        ]
    ).partial(format_instructions=output_parser.get_format_instructions())

    return (
        prompt
        | llm
        | output_parser
    ).with_config(tags=["extraction_chain"])