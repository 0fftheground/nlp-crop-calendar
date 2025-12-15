"""
NLP应用包初始化
"""

from src.app import NLPApp
from src.intent_recognition_manager import IntentRecognitionManager
from src.api_caller import APICaller
from src.agent import NLPAgent, MultiTurnAgent
from src.llm_config import get_llm, LLMConfig
from src.agent_tools import get_all_tools

__version__ = "2.0.0"
__all__ = [
    "NLPApp",
    "IntentRecognitionManager",
    "APICaller",
    "NLPAgent",
    "MultiTurnAgent",
    "get_llm",
    "LLMConfig",
    "get_all_tools",
]
