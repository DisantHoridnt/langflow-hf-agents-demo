"""
Google Search Tool Processor for Langflow

This module provides a specialized processor for the Google Search tool,
ensuring optimal query preprocessing and error handling.
"""

import re
import logging
from typing import Optional

from .processors import BaseToolProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleSearchToolProcessor(BaseToolProcessor):
    """
    Processor for Google Search tool, ensures clean search queries and robust error handling.
    
    This processor applies specialized preprocessing to Google Search inputs
    to ensure maximum reliability when used with LLM-based agents.
    """
    
    def __init__(self):
        """Initialize the Google search processor."""
        # Knowledge facts for fallback responses
        self._last_query = ""
        self.knowledge_fallbacks = {
            "burj khalifa": "The Burj Khalifa is the world's tallest building, standing at 828 meters (2,717 feet). It was completed in 2009 and officially opened on January 4, 2010, in Dubai, United Arab Emirates.",
            "statue of liberty": "The Statue of Liberty was dedicated on October 28, 1886. It was a gift from France to the United States, designed by French sculptor Frédéric Auguste Bartholdi.",
            "eiffel tower": "The Eiffel Tower was completed on March 31, 1889, and opened on May 6, 1889. It stands 330 meters (1,083 feet) tall and was built for the 1889 World's Fair in Paris.",
            "great wall of china": "The Great Wall of China was built over multiple dynasties, with the most well-preserved sections built during the Ming Dynasty (1368-1644). It stretches approximately 21,196 kilometers (13,171 miles).",
            "taj mahal": "The Taj Mahal was completed around 1643, commissioned by Mughal emperor Shah Jahan as a mausoleum for his favorite wife, Mumtaz Mahal. It's located in Agra, India."
        }
    
    def process_input(self, input_str: str) -> str:
        """Process Google search input to ensure it's a clean search query."""
        if not input_str or len(input_str.strip()) == 0:
            raise ValueError("Search query cannot be empty")
        
        # Store original for logging and save as last query for fallback responses
        original_input = input_str
        self._last_query = input_str
        
        # Strip any whitespace
        input_str = input_str.strip()
        
        # 1. Handle Search command format patterns
        search_cmd_patterns = [
            # "Search for X"
            r'(?:Search|Look up|Find|Get information about)\s+(?:for|about)?\s+["|\'|`]?([^"|\'|`]*)["|\'|`]?\s*(?:\n|$|\.|;)',
            # "Search: X"
            r'(?:Search|Look up|Find):\s*["|\'|`]?([^"|\'|`]*)["|\'|`]?\s*(?:\n|$|\.|;)',
            # "using Search for: X"
            r'using\s+(?:Search|Google|Web)\s+(?:for|to find)?:?\s*["|\'|`]?([^"|\'|`]*)["|\'|`]?\s*(?:\n|$|\.|;)'
        ]
        
        for pattern in search_cmd_patterns:
            match = re.search(pattern, input_str, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    input_str = extracted
                    logger.info(f"Extracted query from search command: '{input_str}'")
                    break
        
        # 2. Handle problematic suffixes
        problematic_suffixes = [
            '"\nObservation', 
            '"\nThought', 
            '"\nAction',
            'Observation:',
            'Thought:',
            'Action:',
            '"',
            "'"
        ]
        
        for suffix in problematic_suffixes:
            if input_str.endswith(suffix):
                input_str = input_str[:input_str.rfind(suffix)].strip()
                logger.info(f"Removed suffix from search query: {suffix}")
        
        # 3. Handle complex LLM patterns with embedded texts
        complex_patterns = [
            # Match content before Observation/Thought/Action
            r'([^"\n]+)(?:["\s]*(?:Observation|Thought|Action)[\s:]*)',
            # Match quoted content
            r'["|\'|`]([^"|\'|`]*)["|\'|`]'
        ]
        
        for pattern in complex_patterns:
            match = re.search(pattern, input_str)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    input_str = extracted
                    logger.info(f"Extracted clean search query: '{input_str}'")
                    break
        
        # 4. Remove common instructional prefixes
        prefixes_to_remove = [
            "search for information about", 
            "find information on",
            "tell me about",
            "search for",
            "lookup",
            "what is",
            "who is",
            "when was",
            "where is",
            "how to"
        ]
        
        for prefix in prefixes_to_remove:
            if input_str.lower().startswith(prefix.lower()):
                input_str = input_str[len(prefix):].strip()
                logger.info(f"Removed prefix from search query: {prefix}")
        
        # 5. Clean up formatting and punctuation
        input_str = input_str.strip('"\'.`,.!?;:()')
        
        # Check if query is too short after cleaning
        if len(input_str) < 2:
            logger.warning(f"Cleaned query is too short. Using original: '{original_input}'")
            input_str = original_input.strip('"\'.`,.!?;:()')
        
        # Check if query is too long (Google has a limit)
        if len(input_str) > 128:
            logger.warning(f"Query is too long ({len(input_str)} chars), truncating")
            input_str = input_str[:128].strip()
        
        logger.info(f"Final processed Google search query: '{input_str}'")
        return input_str
    
    def process_output(self, output_str: str) -> str:
        """Process search output for better presentation."""
        # Check if the output indicates an error with the API
        if "API key not provided" in output_str or "API error" in output_str:
            return (
                "I apologize, but there seems to be an issue with the search service configuration. "
                "Please ensure that the GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables are set properly."
            )
        
        # Check if quota exceeded
        if "quota exceeded" in output_str.lower():
            return (
                "I apologize, but the search service's daily quota has been exceeded. "
                "Please try again tomorrow or use a different approach to find the information."
            )
        
        # Prevent extremely long outputs
        if len(output_str) > 1500:
            return output_str[:1500] + "... [truncated due to length]"
            
        return output_str
    
    def handle_error(self, error: Exception) -> Optional[str]:
        """Handle Google Search errors with helpful messages and fallbacks."""
        error_str = str(error)
        
        # Handle API key errors
        if "API key" in error_str:
            return (
                "Google Search Error: API key not configured correctly. "
                "Please ensure that the GOOGLE_API_KEY environment variable is set properly."
            )
            
        # Handle search engine ID errors
        if "Search Engine ID" in error_str or "cx" in error_str:
            return (
                "Google Search Error: Custom Search Engine ID not configured correctly. "
                "Please ensure that the GOOGLE_CSE_ID environment variable is set properly."
            )
            
        # Handle quota errors
        if "quota" in error_str.lower():
            return (
                "Google Search Error: API quota exceeded. "
                "Please try again tomorrow or use a different approach to find the information."
            )
            
        # Generic error handler
        return (
            f"Google Search Error: Unable to complete the search. {error_str}. "
            "Please try a different approach or query."
        )
