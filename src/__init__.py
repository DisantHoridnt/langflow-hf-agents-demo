"""Hugging Face Native Agents

This package provides agents built on LangChain that work with any LLM,
including open-source models hosted on Hugging Face.
"""

__version__ = "0.1.0"

# Import the standalone agent implementations
from .agents import StandaloneReActAgent, StandalonePlanExecuteAgent

# Conditionally import Langflow components if available
try:
    from .langflow_components import ReActAgentComponent, PlanExecuteAgentComponent
except ImportError:
    # Langflow is not installed, which is fine for standalone usage
    pass
