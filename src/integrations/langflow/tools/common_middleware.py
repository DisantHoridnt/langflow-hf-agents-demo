"""
Common middleware functions that apply to all tools.

This module provides middleware functions that can be registered globally to apply
to all tools, regardless of their type.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Pattern, Union, Type

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_action_format(tool: Any, input_str: str) -> str:
    """
    Clean up common formatting issues in tool inputs.
    
    This middleware removes prefixes like "Action Input:" that might have
    been included in the tool input by the LLM.
    
    Args:
        tool: The tool being called
        input_str: The input string from the LLM
        
    Returns:
        Cleaned input string
    """
    # Strip any whitespace
    cleaned_input = input_str.strip()
    
    # Remove any "Action Input:" prefix that might have been included
    if cleaned_input.lower().startswith("action input:"):
        cleaned_input = cleaned_input[len("action input:"):].strip()
        
    # Remove any tool name prefix that might have been included
    tool_prefix = f"{tool.name}:"
    if cleaned_input.lower().startswith(tool_prefix.lower()):
        cleaned_input = cleaned_input[len(tool_prefix):].strip()
        
    return cleaned_input


def log_tool_usage(tool: Any, input_str: str) -> str:
    """
    Log tool usage for monitoring and debugging.
    
    Args:
        tool: The tool being called
        input_str: The input string from the LLM
        
    Returns:
        The unchanged input string
    """
    logger.info(f"Tool called: {tool.name} with input: '{input_str}'")
    return input_str


def format_tool_error(tool: Any, error: Exception) -> Optional[str]:
    """
    Format error messages from tools consistently.
    
    Args:
        tool: The tool that raised the error
        error: The exception that was raised
        
    Returns:
        Formatted error message, or None to let other handlers try
    """
    # Check for common error patterns
    error_str = str(error)
    
    if "invalid format" in error_str.lower():
        return (
            f"Error using {tool.name}: The input format was invalid. "
            "Please provide a properly formatted input for this tool."
        )
    
    if "access denied" in error_str.lower() or "permission" in error_str.lower():
        return (
            f"Error using {tool.name}: Access denied. "
            "This tool may require specific permissions or authentication."
        )
    
    # For other errors, let other handlers try
    return None


def register_global_middleware(middleware_manager):
    """
    Register global middleware with the middleware manager.
    
    Args:
        middleware_manager: The middleware manager to register with
    """
    # Register pre-processors
    middleware_manager.register_pre_processor(clean_action_format)
    middleware_manager.register_pre_processor(log_tool_usage)
    
    # Register error handlers
    middleware_manager.register_error_handler(format_tool_error)
    
    logger.info("Registered global middleware functions")
