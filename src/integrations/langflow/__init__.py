"""Custom Langflow components.

This module contains custom Langflow components for building agents.
We use an adapter pattern to support different versions of Langflow.
"""

import logging
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import our adapter module first
from .adapters import LCToolsAgentComponent, Component

# Set up exports list
__all__ = ["ReActAgentComponent", "PlanExecuteAgentComponent", "LCToolsAgentComponent"]

# Try to import components with proper error handling
try:
    from .custom_react_agent import ReActAgentComponent
    from .plan_execute_component import PlanExecuteAgentComponent
    logger.info("Successfully imported Langflow components")
except ImportError as e:
    logger.warning(f"Could not import components: {str(e)}.\nCreating mock versions instead.")
    
    # Create mock implementations as fallbacks
    class ReActAgentComponent(LCToolsAgentComponent):
        """Mock implementation of the ReAct agent component."""
        display_name = "ReAct Agent"
        description = "ReAct pattern agent that uses any LLM including open-source models."
        icon = "🤔"
        
        def build_agent(self):
            """Build the agent."""
            return None
    
    class PlanExecuteAgentComponent(LCToolsAgentComponent):
        """Mock implementation of the Plan-Execute agent component."""
        display_name = "Plan-Execute Agent"
        description = "Plan-and-Execute agent that uses any LLM including open-source models."
        icon = "🗺️"
        
        def build_agent(self):
            """Build the agent."""
            return None
