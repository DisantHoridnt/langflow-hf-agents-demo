"""
Hugging Face Native Agent Components for Langflow.

This module provides components that work with Langflow:
1. ReActAgentComponent - Uses the ReAct (Reasoning and Acting) pattern
2. PlanExecuteAgentComponent - Uses the Plan-and-Execute pattern

Both components leverage LangChain's agent frameworks but don't require tool-calling
capabilities from the LLM, making them suitable for open-source models.
"""

from .react_agent import ReActAgentComponent
from .plan_execute_agent import PlanExecuteAgentComponent

__all__ = ["ReActAgentComponent", "PlanExecuteAgentComponent"]
