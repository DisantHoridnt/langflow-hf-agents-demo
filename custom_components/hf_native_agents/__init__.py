"""
Hugging Face Native Agent Components for Langflow.

This module provides two agent components that work with any LLM (including HF-hosted models):
1. ReActAgentComponent - Uses the ReAct (Reasoning and Acting) pattern
2. PlanExecuteAgentComponent - Uses the Plan-and-Execute pattern

Both components leverage LangChain's agent frameworks but don't require tool-calling
capabilities from the LLM, making them suitable for open-source models.
"""

from langflow.custom import CustomComponent

from .react_agent import ReActAgentComponent
from .plan_execute_agent import PlanExecuteAgentComponent

__all__ = ["ReActAgentComponent", "PlanExecuteAgentComponent"]
