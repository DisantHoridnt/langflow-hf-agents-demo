"""
Integration module for the tool handling system.

This module provides functions to easily integrate the tool handling system
with existing Langflow components.
"""

import logging
from typing import List, Any, Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def enhance_tools_for_agent(tools: List[Any], debug: bool = False) -> List[Any]:
    """
    Enhance a list of tools with middleware for use by an agent.
    
    This function applies the appropriate middleware to each tool based on its type,
    making them more robust and error-resistant when used by LLM-based agents.
    
    Args:
        tools: A list of tools to enhance
        debug: If True, enables more verbose logging
        
    Returns:
        A list of enhanced tools
        
    Example:
        ```python
        from langchain.agents import load_tools
        from langflow.tools.integration import enhance_tools_for_agent
        
        # Load standard tools
        tools = load_tools(["wikipedia", "llm-math", "ddg-search"])
        
        # Enhance them with middleware
        enhanced_tools = enhance_tools_for_agent(tools)
        
        # Use the enhanced tools with your agent
        agent = create_react_agent(llm, enhanced_tools, prompt)
        ```
    """
    if debug:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
    
    # Import the registry and middleware manager
    from . import tool_registry, middleware_manager
    
    # Register global middleware if needed
    from .common_middleware import register_global_middleware
    register_global_middleware(middleware_manager)
    
    # Enhance the tools
    enhanced_tools = middleware_manager.enhance_tools(tools, tool_registry)
    
    logger.info(f"Enhanced {len(enhanced_tools)} tools with appropriate middleware")
    
    if debug:
        # Log registered tools
        registered_tools = tool_registry.list_registered_tools()
        logger.debug("Registered tool processors:")
        for processor_name, tool_names in registered_tools.items():
            logger.debug(f"  {processor_name}: {', '.join(tool_names)}")
    
    return enhanced_tools


def register_custom_tool_processor(processor_class: Any, 
                                  tool_names: Optional[List[str]] = None,
                                  patterns: Optional[List[str]] = None,
                                  categories: Optional[List[str]] = None,
                                  is_default: bool = False) -> None:
    """
    Register a custom tool processor with the registry.
    
    This function allows users to add their own specialized tool processors
    to handle specific types of tools.
    
    Args:
        processor_class: The processor class to register
        tool_names: List of exact tool names this processor handles
        patterns: List of regex patterns to match tool names
        categories: List of tool categories this processor handles
        is_default: If True, sets this processor as the default
        
    Example:
        ```python
        from langflow.tools.processors import BaseToolProcessor
        from langflow.tools.integration import register_custom_tool_processor
        
        class MyCustomToolProcessor(BaseToolProcessor):
            def process_input(self, input_str: str) -> str:
                # Custom input processing logic
                return input_str.strip().lower()
                
            def handle_error(self, error: Exception) -> Optional[str]:
                return f"Custom error handling: {str(error)}"
                
        # Register the processor
        register_custom_tool_processor(
            MyCustomToolProcessor,
            tool_names=["my-custom-tool"],
            patterns=[r".*custom.*"]
        )
        ```
    """
    # Import the registry
    from . import tool_registry
    
    # Register the processor
    tool_registry.register_processor(
        processor_class,
        tool_names=tool_names,
        patterns=patterns,
        categories=categories,
        is_default=is_default
    )
    
    logger.info(f"Registered custom tool processor: {processor_class.__name__}")
