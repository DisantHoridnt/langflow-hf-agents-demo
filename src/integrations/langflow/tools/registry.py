"""
ToolRegistry - Central registry for tool processors and middleware.

This module provides a scalable approach to registering and managing tool processing behaviors,
allowing for easy addition of new tools and specialized handling logic.
"""

import logging
from typing import Dict, Type, List, Optional, Any, Callable, Set
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for tool processors.
    
    This class manages the registration of tool processors by name, pattern, or category,
    and provides methods to retrieve the appropriate processor for a given tool.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        # Store processors by exact name (case-insensitive)
        self._processors_by_name: Dict[str, Any] = {}
        
        # Store processors by regex pattern
        self._processors_by_pattern: List[tuple] = []
        
        # Store processors by tool category
        self._processors_by_category: Dict[str, Any] = {}
        
        # Default processor to use when no specific processor is found
        self._default_processor = None
    
    def register_processor(self, processor_class: Type, 
                          tool_names: Optional[List[str]] = None,
                          patterns: Optional[List[str]] = None,
                          categories: Optional[List[str]] = None,
                          is_default: bool = False) -> None:
        """
        Register a processor for specific tool names, patterns, or categories.
        
        Args:
            processor_class: The processor class to register
            tool_names: List of exact tool names this processor handles
            patterns: List of regex patterns to match tool names
            categories: List of tool categories this processor handles
            is_default: If True, sets this processor as the default
        """
        if tool_names:
            for name in tool_names:
                self._processors_by_name[name.lower()] = processor_class
                logger.info(f"Registered processor {processor_class.__name__} for tool '{name}'")
        
        if patterns:
            for pattern in patterns:
                compiled_pattern = re.compile(pattern, re.IGNORECASE)
                self._processors_by_pattern.append((compiled_pattern, processor_class))
                logger.info(f"Registered processor {processor_class.__name__} for pattern '{pattern}'")
        
        if categories:
            for category in categories:
                self._processors_by_category[category.lower()] = processor_class
                logger.info(f"Registered processor {processor_class.__name__} for category '{category}'")
        
        if is_default:
            self._default_processor = processor_class
            logger.info(f"Set {processor_class.__name__} as default processor")
    
    def get_processor_for_tool(self, tool: Any) -> Any:
        """
        Get the appropriate processor for a given tool.
        
        Args:
            tool: The tool object to find a processor for
            
        Returns:
            A processor instance appropriate for the tool
        """
        tool_name = getattr(tool, "name", str(tool)).lower()
        
        # Check for exact name match first (most specific)
        if tool_name in self._processors_by_name:
            processor_class = self._processors_by_name[tool_name]
            return processor_class()
        
        # Check for pattern matches
        for pattern, processor_class in self._processors_by_pattern:
            if pattern.search(tool_name):
                return processor_class()
        
        # Check for category matches if tool has a category attribute
        tool_category = getattr(tool, "category", "").lower()
        if tool_category and tool_category in self._processors_by_category:
            processor_class = self._processors_by_category[tool_category]
            return processor_class()
        
        # Fall back to default processor
        if self._default_processor:
            return self._default_processor()
        
        # If we get here, we have no processor for this tool
        logger.warning(f"No processor found for tool '{tool_name}', and no default processor set")
        raise ValueError(f"No processor found for tool '{tool_name}'")
    
    def list_registered_tools(self) -> Dict[str, Set[str]]:
        """
        List all registered tools and their processors.
        
        Returns:
            A dictionary mapping processor names to sets of tool names
        """
        result = {}
        
        # Add exact name matches
        for tool_name, processor_class in self._processors_by_name.items():
            processor_name = processor_class.__name__
            if processor_name not in result:
                result[processor_name] = set()
            result[processor_name].add(tool_name)
        
        # Add pattern matches (just show the patterns)
        for pattern, processor_class in self._processors_by_pattern:
            processor_name = processor_class.__name__
            if processor_name not in result:
                result[processor_name] = set()
            result[processor_name].add(f"pattern:{pattern.pattern}")
        
        # Add category matches
        for category, processor_class in self._processors_by_category.items():
            processor_name = processor_class.__name__
            if processor_name not in result:
                result[processor_name] = set()
            result[processor_name].add(f"category:{category}")
        
        # Add default processor
        if self._default_processor:
            processor_name = self._default_processor.__name__
            if processor_name not in result:
                result[processor_name] = set()
            result[processor_name].add("default")
        
        return result
