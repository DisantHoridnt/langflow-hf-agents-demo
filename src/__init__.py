"""Hugging Face Native Agents

Professional implementation of agents built on LangChain that work with any LLM,
including open-source models hosted on Hugging Face.
"""

__version__ = "0.1.0"

# Direct imports for convenience
from .core.agents import StandaloneReActAgent, StandalonePlanExecuteAgent

# Conditionally import Langflow components if available
try:
    from .integrations.langflow import ReActAgentComponent, PlanExecuteAgentComponent
except ImportError:
    # Langflow is not installed, which is fine for standalone usage
    pass

# Expose submodules for more granular imports
from . import core
from . import integrations
