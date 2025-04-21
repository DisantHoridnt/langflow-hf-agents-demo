"""
Simple test for Google Search Tool

This script directly tests the Google Search tool without requiring
other LangChain dependencies.
"""

import os
import sys
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our Google Search tool
from src.integrations.langflow.tools.google_search import GoogleSearchTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run a simple test of the Google Search tool."""
    # Check if API keys are set
    if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("GOOGLE_CSE_ID"):
        logger.error(
            "Please set the GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables."
            "\nYou can get these from the Google Cloud Console and Programmable Search Engine."
        )
        return
    
    # Initialize the Google Search tool
    search_tool = GoogleSearchTool()
    
    # Test queries
    test_queries = [
        "What is the capital of Japan?",
        "Who won the 2023 FIFA Women's World Cup?",
        "What is the current population of New York City?",
        "When was the first iPhone released?",
        "What are the latest developments in AI?"
    ]
    
    # Run searches
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n===== TEST QUERY {i}: {query} =====")
        try:
            result = search_tool._run(query)
            print(result)
        except Exception as e:
            logger.error(f"Error on query '{query}': {str(e)}")

if __name__ == "__main__":
    main()
