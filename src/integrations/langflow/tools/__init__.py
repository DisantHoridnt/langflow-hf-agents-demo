"""
Tools package for Langflow integration.

This package provides a scalable system for tool registration, middleware, and utility functions
to enhance the robustness of LLM-based agents when working with tools.
"""

from .registry import ToolRegistry
from .middleware import MiddlewareManager
from .processors import (
    register_default_processors,
    BaseToolProcessor,
    SearchToolProcessor,
    CalculatorToolProcessor,
    WikipediaToolProcessor,
    DefaultToolProcessor
)

# Import Google Search components
from .google_processor import GoogleSearchToolProcessor
from .google_search import get_google_search_tool, google_search

# Initialize the global registry and middleware manager
tool_registry = ToolRegistry()
middleware_manager = MiddlewareManager()

# Register the default tool processors
register_default_processors(tool_registry)

__all__ = [
    "tool_registry",
    "middleware_manager",
    "BaseToolProcessor",
    "SearchToolProcessor",
    "CalculatorToolProcessor",
    "WikipediaToolProcessor",
    "DefaultToolProcessor",
    "ToolRegistry",
    "MiddlewareManager",
    "GoogleSearchToolProcessor",
    "GoogleSearchTool",
    "get_google_search_tool",
]
