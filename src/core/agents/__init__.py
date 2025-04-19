"""Agent implementations for Hugging Face models.

This module provides self-contained agent implementations that work with
any LLM, including open-source models from Hugging Face.
"""

from .react import StandaloneReActAgent
from .plan_execute import StandalonePlanExecuteAgent

__all__ = ["StandaloneReActAgent", "StandalonePlanExecuteAgent"]
