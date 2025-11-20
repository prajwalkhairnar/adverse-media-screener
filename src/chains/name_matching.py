from typing import Type
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

from config.prompts import NAME_MATCHING_SYSTEM_PROMPT, NAME_MATCHING_USER_PROMPT
from src.models.schemas import NameMatchingOutput

def create_name_matching_chain(llm: BaseLanguageModel) -> Runnable:
    """
    Creates the LangChain Runnable for Name Matching (Section 5.2).

    Args:
        llm: The configured LLM.

    Returns:
        A LangChain Runnable that takes prompt variables and returns a NameMatchingOutput model.
    """
    output_parser = JsonOutputParser(pydantic_object=NameMatchingOutput)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", NAME_MATCHING_SYSTEM_PROMPT),
            ("human", NAME_MATCHING_USER_PROMPT),
        ]
    ).partial(format_instructions=output_parser.get_format_instructions())

    return (
        prompt
        | llm #.with_structured_output(NameMatchingOutput)
        | output_parser
    ).with_config(tags=["name_matching_chain"])