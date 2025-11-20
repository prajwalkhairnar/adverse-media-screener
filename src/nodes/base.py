from typing import Dict, Any, Type, Union, List
from abc import ABC, abstractmethod
import time

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from src.llm.cost_tracker import CostTracker
from src.graph.state import ScreeningState
from src.utils.logger import get_logger
from config.settings import Settings, LLMProvider

logger = get_logger("BaseNode")

class BaseNode(ABC):
    """
    Abstract base class for all LLM-driven nodes in the LangGraph workflow.
    It handles common logic like cost tracking and error handling.
    """

    def __init__(self, llm: BaseLanguageModel, settings: Settings, cost_tracker: CostTracker):
        self.llm = llm
        self.settings = settings
        self.cost_tracker = cost_tracker

    @abstractmethod
    def run(self, state: ScreeningState, llm_provider: LLMProvider) -> Dict[str, Any]:
        """
        The main execution method for the node. Must be implemented by subclasses.
        """
        pass
    
    def _invoke_chain_with_tracking(
        self,
        chain: Runnable,
        input_vars: Dict[str, Any],
        step_name: str,
        llm_provider: LLMProvider,
        llm_model: str,
    ) -> BaseModel:
        """
        Invokes a LangChain Runnable and tracks LLM usage/cost.
        
        Args:
            chain: The configured LangChain Runnable (Prompt | LLM | Parser).
            input_vars: Dictionary of inputs to the chain.
            step_name: The name of the node/step for logging.
            llm_provider: The provider enum.
            llm_model: The model string.
            
        Returns:
            The parsed Pydantic output model.
        """
        
        start_time = time.time()
        
        # 1. Invoke the chain
        response = chain.invoke(input_vars)
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Fallback/Simplification: Estimate token counts based on length
        # In a production system, this would be retrieved from LangChain callbacks.
        input_length = len(str(input_vars).split()) # Simple word count
        output_length = len(str(response).split()) # Simple word count
        
        # Track the usage (simulated/estimated)
        self.cost_tracker.record_usage(
            step_name=step_name,
            provider=llm_provider,
            model_name=llm_model,
            prompt_tokens=input_length * 1.5,
            completion_tokens=output_length * 1.5,
            cache_read_tokens=0,
            cache_write_tokens=0,
            latency_ms=duration_ms,
        )
        
        return response