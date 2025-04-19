"""
Adapter layer for Langflow compatibility.

This module provides compatibility classes and functions to bridge between
different versions of Langflow, allowing our components to work regardless of
the specific Langflow version installed.
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Type, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom field type adapters - create our own versions if not available from Langflow
try:
    from langflow.field_typing import LanguageModel, Tool, BaseMemory
    logger.info("Successfully imported field types from Langflow")
    USING_LANGFLOW_TYPES = True
except ImportError:
    logger.warning("Could not import field types from Langflow - creating adapter types")
    USING_LANGFLOW_TYPES = False
    
    # Create adapter types
    LanguageModel = BaseLanguageModel
    Tool = BaseTool
    # Simple base memory class if not available
    class BaseMemory:
        """Base memory class adapter."""
        pass

# Try to import from different Langflow versions/structures
try:
    # First try the expected structure (newer versions)
    from langflow.base.agents.agent import LCToolsAgentComponent as OriginalLCToolsAgentComponent
    from langflow.custom import Component
    logger.info("Using standard Langflow import structure")
    USING_STANDARD_IMPORTS = True
except ImportError:
    try:
        # Try alternate structure (older or restructured versions)
        from langflow.custom.custom_component import Component
        logger.info("Using alternate Langflow import structure")
        USING_STANDARD_IMPORTS = False
    except ImportError:
        logger.warning("Could not import Langflow components - creating mock versions")
        USING_STANDARD_IMPORTS = False
        
        # Create minimal versions of required classes if imports fail
        class Component:
            """Mock Component base class when Langflow is not available."""
            display_name: str = "Base Component"
            description: str = "Base component for Langflow"
            icon: str = "🔧"
            
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                
            def build(self) -> Any:
                """Build the component."""
                raise NotImplementedError("This is a mock base class")


# Create our own LCToolsAgentComponent if the original isn't available
class LCToolsAgentComponent(Component):
    """
    Base class for tools-based agent components.
    
    This either inherits from the original Langflow class or reimplements
    the essential functionality when the original isn't available.
    """
    
    # Define the base inputs that all tools-based agent components need
    _base_inputs = [
        ("llm", LanguageModel, None),
        ("tools", List[Tool], [])
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tools = []
        self.llm = None
        self.memory = None
        
    def validate_tool_names(self):
        """Validate that tool names are unique."""
        if not hasattr(self, "tools") or not self.tools:
            return
                
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Tool names must be unique")

    def build(self):
        """Build and return the agent."""
        return self.build_agent()
    
    def build_agent(self):
        """Build the agent - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement build_agent")
    
    def set(self, **kwargs):
        """Set component attributes."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
