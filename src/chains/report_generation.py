# src/chains/report_generation.py

"""
LangChain Runnable for final report generation. (Section 5.4)

This chain takes the complete state (assessments, entities, metadata)
and uses a specific prompt to generate the final human-readable report string.
"""
from typing import Dict
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

# Import prompt constants from config
# NOTE: These constants must be defined in config/prompts.py next!
from config.prompts import REPORT_GENERATION_SYSTEM_PROMPT, REPORT_GENERATION_USER_PROMPT


def create_report_generation_chain(llm: BaseLanguageModel) -> Runnable:
    """
    Creates the LangChain Runnable for Report Generation (Section 5.4).

    Args:
        llm: The configured LLM (e.g., ChatOpenAI, ChatAnthropic).

    Returns:
        A LangChain Runnable that takes prompt variables and returns the final report string.
    """
    
    # 1. Define the Prompt Template
    # This template will expect all the state fields passed as prompt variables 
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REPORT_GENERATION_SYSTEM_PROMPT),
            ("human", REPORT_GENERATION_USER_PROMPT),
        ]
    )

    # 2. Define the Chain
    # The chain is simple: Prompt -> LLM -> String Output
    chain = (
        prompt
        | llm
        | StrOutputParser() # The final report is just a string
    ).with_config(tags=["report_generation_chain"])
    
    return chain