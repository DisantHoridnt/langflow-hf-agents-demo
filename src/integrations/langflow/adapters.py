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

# --- Fallback Input Classes (if langflow.io is not available) ---
# These provide basic structure for component input definitions.
class MultilineInput:
    def __init__(self, *, name: str, display_name: str = "", info: str = "", value: str = "", advanced: bool = False):
        self.name = name
        self.display_name = display_name or name
        self.info = info
        self.value = value
        self.advanced = advanced

class IntInput:
    def __init__(self, *, name: str, display_name: str = "", info: str = "", value: int = 0, advanced: bool = False):
        self.name = name
        self.display_name = display_name or name
        self.info = info
        self.value = value
        self.advanced = advanced

class BoolInput:
    def __init__(self, *, name: str, display_name: str = "", info: str = "", value: bool = False, advanced: bool = False):
        self.name = name
        self.display_name = display_name or name
        self.info = info
        self.value = value
        self.advanced = advanced

# Create our own LCToolsAgentComponent if the original isn't available
class LCToolsAgentComponent(Component):
    """
    Adapter class for LCToolsAgentComponent.
    
    This provides a compatible interface with Langflow's agent components, implementing
    the essential functionality when the original isn't available.
    """
    
    # Define the base inputs that all tools-based agent components need using proper Input objects
    # Define a fallback Input class in case Langflow imports fail
    class InputBase:
        """Fallback Input class when Langflow imports fail."""
        def __init__(self, name, type=None, default=None, display_name=None, info=None, advanced=False):
            self.name = name
            self.type = type
            self.default = default
            self.display_name = display_name or name
            self.info = info
            self.advanced = advanced
    
    # Try different import paths for Langflow's Input class
    try:
        from langflow.interface.input_interface import Input
        logger.info("Using langflow.interface.input_interface.Input")
    except ImportError:
        try:
            from langflow.interface.base import Input
            logger.info("Using langflow.interface.base.Input")
        except ImportError:
            try:
                from langflow.custom.customs import Input
                logger.info("Using langflow.custom.customs.Input")
            except ImportError:
                # Final fallback to our own implementation
                logger.warning("Using fallback Input implementation")
                Input = InputBase
        
    _base_inputs = [
        Input(
            name="llm",
            type=LanguageModel,
            default=None,
            display_name="Language Model",
            info="The language model to use for the agent"
        ),
        Input(
            name="tools",
            type=List[Tool],
            default=[],
            display_name="Tools",
            info="The tools available to the agent"
        )
    ]
    
    def __init__(self, **kwargs):
        # Initialize component attributes before super().__init__
        self.tools = []
        self.llm = None
        self.memory = None
        
        # Convert tuple inputs to proper Input objects if needed
        if hasattr(self, '_base_inputs') and isinstance(self._base_inputs, list):
            # Use our already-defined Input class (no new import needed)
            # Check if any input is in tuple format and convert it
            for i, input_def in enumerate(self._base_inputs):
                if isinstance(input_def, tuple):
                    # Convert tuple to Input object using InputBase directly to avoid circular reference
                    self._base_inputs[i] = self.InputBase(
                        name=input_def[0],
                        type=input_def[1],
                        default=input_def[2]
                    )
        
        # Call super().__init__ after fixing inputs
        super().__init__(**kwargs)
        
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
