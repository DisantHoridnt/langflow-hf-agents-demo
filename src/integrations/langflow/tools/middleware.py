"""
Middleware Manager for Tool Processing

This module provides a middleware system that can be applied to agent tools
to enhance their robustness, standardize error handling, and provide consistent
input/output formatting.
"""

import logging
from typing import Any, List, Dict, Callable, Optional, Type, Union
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MiddlewareManager:
    """
    Manages the application of middleware to tools.
    
    This class provides methods to apply registered middleware to tools,
    with support for pre-processing inputs and post-processing outputs.
    """
    
    def __init__(self):
        """Initialize an empty middleware manager."""
        # Middleware functions for pre-processing inputs
        self._pre_processors: List[Callable] = []
        
        # Middleware functions for post-processing outputs
        self._post_processors: List[Callable] = []
        
        # Middleware functions for error handling
        self._error_handlers: List[Callable] = []
    
    def register_pre_processor(self, processor: Callable) -> None:
        """
        Register a pre-processor function.
        
        Pre-processors are called before tool execution and can modify 
        the input to the tool.
        
        Args:
            processor: Function that takes (tool, input) and returns modified input
        """
        self._pre_processors.append(processor)
        logger.info(f"Registered pre-processor: {processor.__name__}")
    
    def register_post_processor(self, processor: Callable) -> None:
        """
        Register a post-processor function.
        
        Post-processors are called after tool execution and can modify
        the output from the tool.
        
        Args:
            processor: Function that takes (tool, output) and returns modified output
        """
        self._post_processors.append(processor)
        logger.info(f"Registered post-processor: {processor.__name__}")
    
    def register_error_handler(self, handler: Callable) -> None:
        """
        Register an error handler function.
        
        Error handlers are called when a tool execution raises an exception.
        
        Args:
            handler: Function that takes (tool, error) and returns a response
        """
        self._error_handlers.append(handler)
        logger.info(f"Registered error handler: {handler.__name__}")
    
    def apply_middleware(self, tool: Any, processor: Any) -> Any:
        """
        Apply middleware to a tool.
        
        This wraps the tool's _run method with pre-processors, post-processors,
        and error handlers.
        
        Args:
            tool: The tool to apply middleware to
            processor: The tool processor to use for specialized handling
            
        Returns:
            A new tool with middleware applied
        """
        from langchain_core.tools import BaseTool, Tool
        
        # Get the original run function
        original_run = tool._run
        
        # Create a wrapper function
        @wraps(original_run)
        def run_with_middleware(self, input_str: str, **kwargs) -> str:
            """Run the tool with middleware applied."""
            # Apply specialized processor pre-processing
            try:
                processed_input = processor.process_input(input_str)
            except Exception as e:
                logger.warning(f"Error in tool processor input handling: {str(e)}")
                processed_input = input_str
                
            # Apply general pre-processors
            for pre_processor in self._pre_processors:
                try:
                    processed_input = pre_processor(tool, processed_input)
                except Exception as e:
                    logger.warning(f"Error in pre-processor {pre_processor.__name__}: {str(e)}")
            
            # Run the original function
            try:
                result = original_run(self, processed_input, **kwargs)
                
                # Apply processor-specific output handling
                try:
                    result = processor.process_output(result)
                except Exception as e:
                    logger.warning(f"Error in tool processor output handling: {str(e)}")
                
                # Apply general post-processors
                for post_processor in self._post_processors:
                    try:
                        result = post_processor(tool, result)
                    except Exception as e:
                        logger.warning(f"Error in post-processor {post_processor.__name__}: {str(e)}")
                
                return result
                
            except Exception as e:
                logger.warning(f"Error in tool execution: {str(e)}")
                
                # Special handler for the DuckDuckGo results error
                if "cannot access local variable 'results'" in str(e):
                    logger.warning("Detected DuckDuckGo results error, providing fallback response")
                    return (
                        "I apologize, but the search service is currently unavailable. " 
                        "Let me answer based on my general knowledge instead.\n\n"
                        "For mathematical questions, I recommend using the Calculator tool directly."
                    )
                
                # Apply processor-specific error handling
                try:
                    result = processor.handle_error(e)
                    if result:
                        return result
                except Exception as nested_e:
                    logger.warning(f"Error in processor error handler: {str(nested_e)}")
                
                # Apply general error handlers
                for error_handler in self._error_handlers:
                    try:
                        result = error_handler(tool, e)
                        if result:
                            return result
                    except Exception as nested_e:
                        logger.warning(f"Error in error handler {error_handler.__name__}: {str(nested_e)}")
                
                # Default error response if no handler worked
                return f"Error using {tool.name}: {str(e)}. Please try a different approach."
        
        # Create a new tool with the wrapped function
        enhanced_tool = Tool(
            name=tool.name,
            description=tool.description,
            func=run_with_middleware,
            coroutine=tool._arun,
            return_direct=tool.return_direct
        )
        
        return enhanced_tool
    
    def enhance_tools(self, tools: List[Any], registry: Any) -> List[Any]:
        """
        Enhance a list of tools with middleware.
        
        Args:
            tools: List of tools to enhance
            registry: The ToolRegistry to use for finding processors
            
        Returns:
            List of enhanced tools
        """
        enhanced_tools = []
        
        for tool in tools:
            # Get the appropriate processor from the registry
            processor = registry.get_processor_for_tool(tool)
            
            # Apply middleware to the tool
            enhanced_tool = self.apply_middleware(tool, processor)
            
            enhanced_tools.append(enhanced_tool)
            logger.info(f"Enhanced tool {tool.name} with middleware using {processor.__class__.__name__}")
        
        return enhanced_tools
