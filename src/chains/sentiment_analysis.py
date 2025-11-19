from typing import Type
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

from config.prompts import SENTIMENT_ANALYSIS_SYSTEM_PROMPT, SENTIMENT_ANALYSIS_USER_PROMPT
from src.models.schemas import SentimentOutput

def create_sentiment_analysis_chain(llm: BaseLanguageModel) -> Runnable:
    """
    Creates the LangChain Runnable for Sentiment Analysis (Section 5.3).

    Args:
        llm: The configured LLM.

    Returns:
        A LangChain Runnable that takes prompt variables and returns a SentimentOutput model.
    """
    output_parser = JsonOutputParser(pydantic_object=SentimentOutput)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SENTIMENT_ANALYSIS_SYSTEM_PROMPT),
            ("human", SENTIMENT_ANALYSIS_USER_PROMPT),
        ]
    ).partial(format_instructions=output_parser.get_format_instructions())

    return (
        prompt
        | llm.with_structured_output(SentimentOutput)
        | output_parser
    ).with_config(tags=["sentiment_analysis_chain"])