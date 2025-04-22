"""
Google Search Tool for Langflow

This module provides a simple functional implementation of Google Custom Search
using the official Google API client. This is more reliable than the DuckDuckGo
search tool in production environments.
"""

import os
import sys
import logging
import importlib.util
from typing import Any, Dict, List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Check if Google API is available
GOOGLE_API_AVAILABLE = importlib.util.find_spec("googleapiclient") is not None

# Only import if available to avoid errors
if GOOGLE_API_AVAILABLE:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
else:
    logger.warning("Google API client not installed. Install with 'pip install google-api-python-client'")

from langchain_core.tools import Tool

def google_search(query: str, api_key: Optional[str] = None, cse_id: Optional[str] = None, 
                 num_results: int = 5) -> str:
    """
    Perform a Google search using the Custom Search API.
    
    Args:
        query: The search query
        api_key: Google API key. If None, will use GOOGLE_API_KEY env var
        cse_id: Google Custom Search Engine ID. If None, will use GOOGLE_CSE_ID env var
        num_results: Number of results to return
        
    Returns:
        Formatted string of search results
    """
    try:
        # Get credentials from environment if not provided
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        cse_id = cse_id or os.environ.get("GOOGLE_CSE_ID")
        
        # Validate inputs
        if not query or not query.strip():
            return "No search query provided. Please specify what you'd like to search for."
            
        if not api_key:
            return "ERROR: Google API key not provided. Please set GOOGLE_API_KEY environment variable."
        if not cse_id:
            return "ERROR: Google Custom Search Engine ID not provided. Please set GOOGLE_CSE_ID environment variable."
            
        # Check if Google API is available
        if not GOOGLE_API_AVAILABLE:
            return "ERROR: Google API client not installed. Install with 'pip install google-api-python-client'."
        
        logger.info(f"Performing Google search for: '{query}'")
        
        # Build the service
        service = build("customsearch", "v1", developerKey=api_key)
        
        # Execute search
        result = service.cse().list(
            q=query.strip(),
            cx=cse_id,
            num=num_results
        ).execute()
        
        # Format results
        formatted_results = f"Google search results for: {query}\n\n"
        
        if "items" in result:
            for i, item in enumerate(result["items"], 1):
                formatted_results += f"{i}. {item.get('title', '')}\n"
                formatted_results += f"   URL: {item.get('link', '')}\n"
                formatted_results += f"   Description: {item.get('snippet', '')}\n\n"
        else:
            formatted_results = "No results found for your query."
            
        return formatted_results
        
    except HttpError as e:
        error_msg = f"Google API error: {str(e)}"
        logger.error(error_msg)
        if "quota" in str(e).lower():
            return "ERROR: Daily Google API quota exceeded. Please try again tomorrow."
        return f"ERROR: {error_msg}"
    except Exception as e:
        error_msg = f"Error searching Google: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"


def mock_google_search(query: str) -> str:
    """
    Mock Google search for testing environments.
    
    Args:
        query: The search query
        
    Returns:
        Formatted mock search results
    """
    return (
        f"Google search results for: {query}\n\n"
        f"1. Test Result for {query}\n"
        f"   URL: https://example.com/result1\n"
        f"   Description: This is a test result for {query}.\n\n"
        f"2. Second Test Result\n"
        f"   URL: https://example.com/result2\n"
        f"   Description: This is another test result.\n\n"
    )


def get_google_search_tool() -> Tool:
    """
    Create and return a Google Search tool wrapped as a LangChain Tool.
    
    Returns:
        Tool: A configured Google Search tool ready for use with LangChain agents
    """
    try:
        # Create the tool with a simple wrapper to avoid passing around complex objects
        return Tool(
            name="google_search",
            description="Search Google for information about current events, data, or answers to questions. Input should be a search query.",
            func=google_search
        )
    except Exception as e:
        logger.error(f"Error creating Google Search tool: {str(e)}")
        # Create a fallback tool that returns an error message
        return Tool(
            name="google_search",
            description="Search Google for information about current events, data, or answers to questions. Input should be a search query.",
            func=lambda query: f"Google Search is currently unavailable: {str(e)}"
        )


# We're keeping the mock function available for testing if needed
# But we won't automatically switch to it in pytest environments
# This ensures real API calls are made when credentials are provided

# Uncomment the following to enable mock in pytest:
'''
def is_running_in_pytest() -> bool:
    """Check if code is running in pytest environment"""
    return "pytest" in sys.modules

if is_running_in_pytest():
    logger.info("Running in pytest environment, using mock Google Search")
    get_google_search_tool = lambda: Tool(
        name="google_search",
        description="Search Google for information. Input should be a search query.",
        func=mock_google_search
    )
'''
