"""Integration tests for Langflow components with Gemma-2B.

This file contains tests for both ReAct and Plan-Execute components using
the lightweight Gemma-2B-Instruct model, suitable for running on M1/M2 Macs.
"""

import os
import pytest
from dotenv import load_dotenv

# Mark these tests as requiring Langflow and potentially slow
pytestmark = [pytest.mark.langflow, pytest.mark.slow]

# Hugging Face and LangChain imports
# Import from langchain_community instead of langchain per deprecation warnings
from langchain_community.llms import HuggingFaceHub
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import BaseTool, Tool
import math
import re

# Create our own calculator tool implementation using only built-in Python functions
class SimpleCalculatorTool(BaseTool):
    """Tool for performing basic mathematical calculations."""
    name: str = "calculator"
    description: str = "Useful for performing mathematical calculations."

    def _run(self, query: str) -> str:
        """Evaluate a mathematical expression safely."""
        try:
            # Clean the input - only allow numbers, basic operators, and some math functions
            # This adds safety compared to raw eval()
            cleaned_query = query.strip()
            
            # Replace common math operations with their Python equivalents
            cleaned_query = cleaned_query.replace('^', '**')
            
            # Check if expression contains only safe characters
            if not re.match(r'^[0-9+\-*/()\s.,\^]+$', cleaned_query):
                return "Error: Expression contains invalid characters. Only basic math operations are supported."
            
            # Evaluate the expression
            result = eval(cleaned_query, {"__builtins__": {}}, {"math": math})
            return str(result)
        except Exception as e:
            return f"Error calculating {query}: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version of calculator."""
        return self._run(query)

# Import our custom components with try/except to handle potential import errors
try:
    from src.integrations.langflow.react_component import ReActAgentComponent
    from src.integrations.langflow.plan_execute_component import PlanExecuteAgentComponent
    langflow_components_available = True
except ImportError:
    # Create mock classes if imports fail
    class ReActAgentComponent:
        def set(self, **kwargs): return self
        def build_agent(self): return None
    
    class PlanExecuteAgentComponent:
        def set(self, **kwargs): return self
        def build_agent(self): return None
        
    langflow_components_available = False


@pytest.fixture(scope="module")
def gemma_llm():
    """Create a Gemma-2B LLM for testing - lightweight for M1 Max."""
    # Load API token from .env file
    load_dotenv()
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    # Skip tests if no API token is available
    if not api_token:
        pytest.skip("No HUGGINGFACEHUB_API_TOKEN found in .env file")
    
    # Initialize the LLM with Gemma-2B-Instruct
    llm = HuggingFaceHub(
        repo_id="google/gemma-2b-it",
        model_kwargs={"temperature": 0.7, "max_length": 512, "max_new_tokens": 512}
    )
    
    return llm


@pytest.fixture(scope="module")
def tools():
    """Create a set of tools for testing agents."""
    # Create Wikipedia tool
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    # Use Tool wrapper to avoid Pydantic errors
    from langchain_core.tools import Tool
    
    wiki_tool = Tool(
        name="Wikipedia",
        description="Useful for looking up information on Wikipedia.",
        func=wiki._run
    )
    
    # Create our simple Calculator tool
    calculator = SimpleCalculatorTool()
    calc_tool = Tool(
        name="Calculator",
        description="Useful for performing mathematical calculations",
        func=calculator._run
    )
    
    return [wiki_tool, calc_tool]


@pytest.mark.skip(reason="Expensive API call, only run manually")
def test_react_component_with_gemma(gemma_llm, tools):
    """Test the ReAct component with Gemma-2B model."""
    # Skip if Langflow components aren't available
    if not langflow_components_available:
        pytest.skip("Langflow components not available")
        
    # Initialize the ReAct component
    component = ReActAgentComponent()
    
    # Configure with required inputs
    component.set(
        llm=gemma_llm,
        tools=tools,
        system_prompt="You are a helpful assistant with tools. Use the tools when needed to answer questions accurately.",
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True
    )
    
    # Build the agent
    agent = component.build_agent()
    
    # Test with a simple query that uses the calculator
    result = agent.invoke("What is 128 * 43?")
    
    # Assertions
    assert result, "Result should not be empty"
    assert "5504" in str(result), "Result should contain the correct calculation (128*43=5504)"


@pytest.mark.skip(reason="Expensive API call, only run manually")
def test_plan_execute_component_with_gemma(gemma_llm, tools):
    """Test the Plan-Execute component with Gemma-2B model."""
    # Skip if Langflow components aren't available
    if not langflow_components_available:
        pytest.skip("Langflow components not available")
        
    # Initialize the Plan-Execute component
    component = PlanExecuteAgentComponent()
    
    # Configure with required inputs
    component.set(
        llm=gemma_llm,
        tools=tools,
        system_prompt="You are a helpful assistant that first plans what to do, then executes the plan step by step.",
        planner_prompt="Let's break down this task into steps. Create a clear and concise plan.",
        verbose=True,
        max_iterations=2,
        max_subtask_iterations=3,
        handle_parsing_errors=True
    )
    
    # Build the agent
    agent = component.build_agent()
    
    # Test with a query that requires planning and execution
    result = agent.invoke("I need to know who the current French president is and when they were born.")
    
    # Basic validation
    assert result, "Result should not be empty"
    # Note: We're not checking for specific content as it depends on the model's output and could change
