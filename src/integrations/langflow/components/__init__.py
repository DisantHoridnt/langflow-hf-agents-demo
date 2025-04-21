"""
Langflow components package.

This package contains custom Langflow components for the application.
"""

from langflow import Custom

# Import components explicitly so they are registered
from .google_search_component import GoogleSearchToolComponent

__all__ = [
    "GoogleSearchToolComponent",
]
