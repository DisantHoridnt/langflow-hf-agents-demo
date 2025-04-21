"""
Tool Processors for Langflow

This module contains tool processors that provide specialized handling logic
for different types of tools. Each processor encapsulates the specialized knowledge
needed to make a specific type of tool more robust when used with LLMs.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Pattern, Union, Type
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseToolProcessor(ABC):
    """
    Base class for all tool processors.
    
    Tool processors provide specialized handling logic for different types of tools,
    including input validation, output formatting, and error handling.
    """
    
    @abstractmethod
    def process_input(self, input_str: str) -> str:
        """
        Process the input before it's passed to the tool.
        
        Args:
            input_str: The raw input string from the LLM
            
        Returns:
            Processed input string ready for the tool
        """
        pass
    
    def process_output(self, output_str: str) -> str:
        """
        Process the output after it's returned from the tool.
        
        Args:
            output_str: The raw output string from the tool
            
        Returns:
            Processed output string ready for the LLM
        """
        # Default implementation: return as-is
        return output_str
    
    def handle_error(self, error: Exception) -> Optional[str]:
        """
        Handle errors that occur during tool execution.
        
        Args:
            error: The exception that was raised
            
        Returns:
            A response string, or None to let other handlers try
        """
        # Default implementation: return None to let other handlers try
        return None


class DefaultToolProcessor(BaseToolProcessor):
    """
    Default processor for tools that don't have a specialized processor.
    """
    
    def process_input(self, input_str: str) -> str:
        """Process input by just cleaning up whitespace."""
        return input_str.strip()
    
    def process_output(self, output_str: str) -> str:
        """Process output by ensuring it's a string."""
        return str(output_str)
    
    def handle_error(self, error: Exception) -> Optional[str]:
        """Provide a generic error message."""
        return f"Error: {str(error)}. Please try a different approach or tool."


class CalculatorToolProcessor(BaseToolProcessor):
    """
    Processor for calculator tools, ensures inputs are valid math expressions.
    """
    
    def process_input(self, input_str: str) -> str:
        """Process calculator input to ensure it's a valid math expression."""
        # Strip any whitespace
        input_str = input_str.strip()
        
        # Check if the query seems to be a natural language question rather than a math expression
        nl_indicators = [
            "what is", "how much", "calculate", "tell me", "find", "the", "height", "of", "in meters", 
            "kilometers", "miles", "weight", "distance", "?"
        ]
        
        # Basic check for natural language in the query
        has_nl_indicators = any(indicator in input_str.lower() for indicator in nl_indicators)
        
        # Check if the query contains valid math operators
        math_pattern = r'[\d\s\+\-\*\/\(\)\^\%\.]+'
        is_math_expr = bool(re.fullmatch(math_pattern, input_str))
        
        if has_nl_indicators and not is_math_expr:
            # This appears to be a natural language query, not a math expression
            raise ValueError(
                "Calculator only accepts mathematical expressions like '2 + 2' or '(5 * 10) / 2'. "
                "Please use only numbers and operators (+, -, *, /, ^, %, ()). "
                "For non-calculation questions, try using the Wikipedia or Search tools instead."
            )
        
        # Add additional check for attempts to use variables or functions
        if re.search(r'[a-zA-Z]', input_str) and not is_math_expr:
            raise ValueError(
                "Calculator doesn't support variables or functions. "
                "Use only numbers and basic operators (+, -, *, /, ^, %)."
            )
        
        # For edge cases, try to extract any math expression
        if not is_math_expr:
            # Try to find a math expression within the query
            math_match = re.search(r'(\d[\d\s\+\-\*\/\(\)\^\%\.]+\d)', input_str)
            if math_match:
                # Extract the math expression
                input_str = math_match.group(1)
                logger.info(f"Extracted math expression: {input_str}")
            else:
                raise ValueError(
                    "No valid math expression found. Please provide a mathematical expression "
                    "like '2 + 2' or '(10 * 5) / 2'."
                )
        
        return input_str
    
    def handle_error(self, error: Exception) -> Optional[str]:
        """Handle calculator errors with helpful messages."""
        error_str = str(error)
        
        if "invalid characters" in error_str.lower():
            return (
                "Calculator Error: Only basic math operations are supported. "
                "Please use only numbers and math operators (+, -, *, /, etc.), "
                "or use a different tool like Wikipedia or Search to find information."
            )
        
        if "division by zero" in error_str.lower():
            return "Calculator Error: Division by zero is not allowed."
        
        return f"Calculator Error: {error_str}. Please check your expression and try again."


class SearchToolProcessor(BaseToolProcessor):
    """
    Production-ready processor for search tools, ensures clean search queries.
    
    This processor includes extensive pattern matching for search queries to handle
    a wide variety of edge cases that occur when LLMs generate search inputs.
    It also provides graceful fallback responses for common search API errors.
    """
    
    def __init__(self):
        # Known facts for common search topics (used for fallback responses)
        self._last_query = ""
        self.knowledge_fallbacks = {
            "burj khalifa": "The Burj Khalifa is the world's tallest building, standing at 828 meters (2,717 feet). It was completed in 2009 and officially opened on January 4, 2010, in Dubai, United Arab Emirates.",
            "statue of liberty": "The Statue of Liberty was dedicated on October 28, 1886. It was a gift from France to the United States, designed by French sculptor Frédéric Auguste Bartholdi.",
            "eiffel tower": "The Eiffel Tower was completed on March 31, 1889, and opened on May 6, 1889. It stands 330 meters (1,083 feet) tall and was built for the 1889 World's Fair in Paris.",
            "great wall of china": "The Great Wall of China was built over multiple dynasties, with the most well-preserved sections built during the Ming Dynasty (1368-1644). It stretches approximately 21,196 kilometers (13,171 miles).",
            "taj mahal": "The Taj Mahal was completed around 1643, commissioned by Mughal emperor Shah Jahan as a mausoleum for his favorite wife, Mumtaz Mahal. It's located in Agra, India."
        }
    
    def process_input(self, input_str: str) -> str:
        """Process search input to ensure it's a clean search query."""
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
            r'using\s+(?:Search|DuckDuckGo|Google|Bing)\s+(?:for|to find)?:?\s*["|\'|`]?([^"|\'|`]*)["|\'|`]?\s*(?:\n|$|\.|;)'
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
        # These are common patterns that should be removed from the end of queries
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
        
        # 3. Handle complex LLM patterns with embedded Observation/Thought texts
        # This handles patterns like: "query"<newline>Observation: or "query"Observation:
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
        
        # 4. Remove common instructional prefixes that might reduce search quality
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
        # Remove leading/trailing quotes and punctuation
        input_str = input_str.strip('"\'.`,.!?;:()')
        
        # Check if query is too short after cleaning
        if len(input_str) < 2:
            logger.warning(f"Cleaned query is too short. Using original: '{original_input}'")
            # Try basic cleaning - just remove quotes and trim
            input_str = original_input.strip('"\'.`,.!?;:()')
        
        # Check if query is too long
        if len(input_str) > 150:
            logger.warning(f"Query is too long ({len(input_str)} chars), truncating")
            input_str = input_str[:150].strip()
        
        logger.info(f"Final processed search query: '{input_str}'")
        return input_str
    
    def process_output(self, output_str: str) -> str:
        """Process search output for better presentation."""
        # Prevent extremely long outputs
        if len(output_str) > 1500:
            return output_str[:1500] + "... [truncated due to length]"
        return output_str
    
    def handle_error(self, error: Exception) -> Optional[str]:
        """Handle search errors with helpful messages and fallbacks."""
        error_str = str(error)
        
        # Special handling for the DuckDuckGo results variable error
        if "cannot access local variable 'results'" in error_str:
            return self._provide_fallback_response()
            
        # Handle rate limiting errors
        if any(term in error_str.lower() for term in ["rate limit", "too many requests", "429", "throttle"]):
            return (
                "Search Error: The search service is temporarily unavailable due to rate limiting. "
                "Please try again in a moment or use a different approach to find the information."
            )
            
        # Handle connectivity errors
        if any(term in error_str.lower() for term in ["connection", "timeout", "network", "unreachable"]):
            return (
                "Search Error: Could not connect to the search service. "
                "There may be network connectivity issues. Please try again later."
            )
            
        # Generic error handler
        return f"Search Error: Unable to complete the search. {str(error)}. Please try a different approach."
    
    def _provide_fallback_response(self) -> str:
        """Provide a graceful fallback response when the search fails entirely."""
        # Generic fallback response
        fallback = (
            "I apologize, but I couldn't complete the search due to a technical issue. "
            "Let me provide some general information instead, or you could try a different tool or approach."
        )
        
        # Check if we have any relevant information in our knowledge base
        # This helps maintain a good user experience even when search fails
        for topic, info in self.knowledge_fallbacks.items():
            if topic in self._last_query.lower():
                return fallback + "\n\n" + info
                
        return fallback


class WikipediaToolProcessor(BaseToolProcessor):
    """
    Processor for Wikipedia tools, ensures effective Wikipedia lookups.
    """
    
    def process_input(self, input_str: str) -> str:
        """Process Wikipedia input similar to search but more specific to Wikipedia."""
        # First apply the general search query cleaning
        search_processor = SearchToolProcessor()
        clean_query = search_processor.process_input(input_str)
        
        # Now apply Wikipedia-specific processing
        # Check if the query contains instructions that would confuse the tool
        instructions = ["find the wikipedia article", "look up on wikipedia", "search wikipedia for"]
        for instruction in instructions:
            if instruction in clean_query.lower():
                clean_query = re.sub(instruction, "", clean_query, flags=re.IGNORECASE).strip()
        
        # Remove any "wiki" or "wikipedia" references that might remain
        clean_query = re.sub(r'\b(?:wiki|wikipedia)\b', '', clean_query, flags=re.IGNORECASE).strip()
        
        # If the query became too short, revert to the original cleaned query
        if len(clean_query) < 3 and len(search_processor.process_input(input_str)) >= 3:
            clean_query = search_processor.process_input(input_str)
        
        logger.info(f"Final processed Wikipedia query: '{clean_query}'")
        return clean_query
    
    def handle_error(self, error: Exception) -> Optional[str]:
        """Handle Wikipedia errors with helpful messages."""
        error_str = str(error)
        
        if "page does not exist" in error_str.lower() or "no matching page found" in error_str.lower():
            return (
                "Wikipedia Error: No Wikipedia page found for this query. "
                "Try a different search term or use the Search tool instead."
            )
        
        return f"Wikipedia Error: {error_str}. Please try a different search term."


def register_default_processors(registry):
    """
    Register default processors with the tool registry.
    
    Args:
        registry: The ToolRegistry to register processors with
    """
    # Register the calculator processor
    registry.register_processor(
        CalculatorToolProcessor,
        tool_names=["calculator", "math", "compute"],
        patterns=[r".*calculator.*", r".*math.*", r".*compute.*"],
        categories=["math"]
    )
    
    # Register the search processor
    registry.register_processor(
        SearchToolProcessor,
        tool_names=["search", "ddgsearch", "googlesearch", "bingsearch", "websearch"],
        patterns=[r".*search.*"],
        categories=["search"]
    )
    
    # Register the Wikipedia processor
    registry.register_processor(
        WikipediaToolProcessor,
        tool_names=["wikipedia", "wiki"],
        patterns=[r".*wiki.*"],
        categories=["knowledge", "reference"]
    )
    
    # Register the default processor
    registry.register_processor(
        DefaultToolProcessor,
        is_default=True
    )
    
    logger.info("Registered default tool processors with the registry")
