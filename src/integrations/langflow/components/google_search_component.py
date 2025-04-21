"""
Langflow component for Google Search.

This module explicitly exposes the Google Search tool as a Langflow component.
"""

from typing import Optional

from langflow import Custom
from langflow.interface.custom.base import CustomComponent
from langflow.interface.custom.component import component

from src.integrations.langflow.tools.google_search import GoogleSearchTool


@component
class GoogleSearchToolComponent(CustomComponent):
    """
    A component that provides Google Search functionality using the Google Custom Search API.
    """
    
    display_name = "Google Search"
    description = "Search the web using Google's Custom Search API"
    
    def build_config(self):
        return {
            "api_key": {
                "display_name": "API Key",
                "info": "Google API Key (defaults to GOOGLE_API_KEY env var)",
                "type": "str",
                "default": "",
                "required": False,
            },
            "search_engine_id": {
                "display_name": "Search Engine ID",
                "info": "Google Custom Search Engine ID (defaults to GOOGLE_CSE_ID env var)",
                "type": "str", 
                "default": "",
                "required": False,
            },
            "max_results": {
                "display_name": "Max Results",
                "info": "Maximum number of search results to return",
                "type": "int",
                "default": 5,
                "required": False,
            },
            "return_direct": {
                "display_name": "Return Direct",
                "info": "Whether to return the direct result",
                "type": "bool",
                "default": False,
                "required": False,
            },
            "code": {"show": False, "value": ""},
        }
    
    def build(
        self,
        api_key: Optional[str] = None,
        search_engine_id: Optional[str] = None,
        max_results: int = 5,
        return_direct: bool = False,
    ) -> GoogleSearchTool:
        """
        Build the Google Search tool.
        
        Args:
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY env var.
            search_engine_id: Search Engine ID. If not provided, uses GOOGLE_CSE_ID env var.
            max_results: Maximum number of search results to return.
            return_direct: Whether to return the direct result.
            
        Returns:
            GoogleSearchTool: The Google Search tool.
        """
        return GoogleSearchTool(
            api_key=api_key,
            search_engine_id=search_engine_id,
            max_results=max_results,
            return_direct=return_direct
        )
