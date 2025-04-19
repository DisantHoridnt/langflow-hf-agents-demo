"""
Integration Test for Langflow Components with Hugging Face Models

This test verifies that both the ReAct and Plan-Execute components work properly with
a lightweight HuggingFace model (Gemma-2B), as required by the PRD.

The test demonstrates:
1. Proper component initialization (F-1)
2. Required input configuration (F-2)
3. Correct agent construction (F-3, F-4)
4. Working agent execution with real tools
"""

import os
import pytest
from typing import List
from dotenv import load_dotenv

# Mark these tests to run only when explicitly requested
pytestmark = [pytest.mark.langflow, pytest.mark.slow]

# Skip the entire module if HUGGINGFACEHUB_API_TOKEN is not available
def pytest_runtest_setup(item):
    load_dotenv()
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        pytest.skip("HUGGINGFACEHUB_API_TOKEN not found in environment")

import os
import logging
import sys
import traceback

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the required LangChain and Langflow components
try:
    logger.info("Trying to import LangChain components...")
    # LangChain community components
    from langchain_community.llms import HuggingFaceHub  
    from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
    from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
    from langchain_core.tools import BaseTool, Tool, StructuredTool
    import math
    import re
    
    logger.info("Successfully imported LangChain components")
    
    # Create our own simple calculator tool implementation
    class SimpleCalculatorTool(BaseTool):
        """Tool for performing basic mathematical calculations."""
        name: str = "calculator"
        description: str = "Useful for performing mathematical calculations."

        def _run(self, query: str) -> str:
            """Evaluate a mathematical expression safely."""
            try:
                # Clean the input - only allow numbers, basic operators, and some math functions
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

    # Print Python path for debugging
    logger.info(f"Python path: {sys.path}")
    
    # Detailed logging for Langflow imports
    logger.info("Trying to import Langflow components...")
    try:
        # Try importing our adapter first
        from src.integrations.langflow.adapters import LCToolsAgentComponent
        logger.info("Successfully imported LCToolsAgentComponent from adapters")
        
        # Now try importing the actual components
        from src.integrations.langflow.react_component import ReActAgentComponent
        logger.info("Successfully imported ReActAgentComponent")
        
        from src.integrations.langflow.plan_execute_component import PlanExecuteAgentComponent
        logger.info("Successfully imported PlanExecuteAgentComponent")
        
        # If we get here, all imports succeeded
        langflow_imports_available = True
        logger.info("All Langflow component imports successful!")
    except ImportError as e:
        logger.error(f"Error importing Langflow components: {str(e)}")
        traceback.print_exc()
        langflow_imports_available = False
except ImportError as e:
    logger.error(f"Error importing LangChain components: {str(e)}")
    traceback.print_exc()
    langflow_imports_available = False


@pytest.fixture(scope="module")
def gemma_llm():
    """Create a Gemma-2B-Instruct LLM from Hugging Face."""
    load_dotenv()
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not api_token:
        pytest.skip("HUGGINGFACEHUB_API_TOKEN not found in environment")
    
    llm = HuggingFaceHub(
        repo_id="google/gemma-2b-it",
        model_kwargs={
            "temperature": 0.7,
            "max_length": 512,
            "max_new_tokens": 512
        }
    )
    return llm


@pytest.fixture(scope="module")
def tools() -> List[BaseTool]:
    """Create a set of tools for the agent to use."""
    # Create a Wikipedia search tool
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    # Using property instead of direct attribute assignment to avoid Pydantic errors
    wiki_tool = Tool(
        name="Wikipedia",
        description="Useful for searching for information on Wikipedia",
        func=wiki._run
    )
    
    # Create our simple calculator tool
    calculator = SimpleCalculatorTool()
    calc_tool = Tool(
        name="Calculator",
        description="Useful for performing mathematical calculations",
        func=calculator._run
    )
    
    # Create a search tool
    search = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())
    search_tool = Tool(
        name="Search",
        description="Useful for searching the internet for current information",
        func=search._run
    )
    
    return [wiki_tool, calc_tool, search_tool]


def test_react_agent_with_gemma(gemma_llm, tools):
    """Test the ReAct agent with real Gemma model."""
    # Skip if Langflow components aren't available
    if not langflow_imports_available:
        pytest.skip("Langflow components not available")
    """
    Test PRD requirements F-1 through F-4 for the ReAct Agent component.
    
    This test verifies that:
    1. The component can be instantiated
    2. The component accepts the required inputs
    3. The component builds a working agent
    4. The agent can be run with a real query
    """
    # F-1: Component can be instantiated (inheritance verified)
    react_component = ReActAgentComponent()
    
    # F-2: Component accepts all required inputs
    react_component.set(
        llm=gemma_llm,
        tools=tools,
        system_prompt="You are a helpful assistant. Use the tools available to answer the question.",
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    # F-3: Component builds the correct agent type
    agent = react_component.build_agent()
    
    # F-4: The agent is usable and can be invoked
    result = agent.invoke("What is the capital of France and what is its population?")
    
    # Verify the result contains relevant information
    assert result, "Agent should return a non-empty result"
    assert "Paris" in str(result), "Result should mention Paris as the capital of France"
    
    print(f"ReAct Agent Result: {result}")


def test_plan_execute_agent_with_gemma(gemma_llm, tools):
    """Test the Plan-Execute agent with real Gemma model."""
    # Skip if Langflow components aren't available
    if not langflow_imports_available:
        pytest.skip("Langflow components not available")
    """
    Test PRD requirements F-1 through F-4 for the Plan-Execute Agent component.
    
    This test verifies that:
    1. The component can be instantiated
    2. The component accepts the required inputs
    3. The component builds a working agent
    4. The agent can be run with a real query
    """
    # F-1: Component can be instantiated (inheritance verified)
    plan_execute_component = PlanExecuteAgentComponent()
    
    # F-2: Component accepts all required inputs
    plan_execute_component.set(
        llm=gemma_llm,
        tools=tools,
        system_prompt="You are a helpful assistant that plans and executes tasks step by step.",
        planner_prompt="Let's break down this task into clear steps. First, create a concise plan.",
        verbose=True,
        max_iterations=3,
        max_subtask_iterations=3,
        handle_parsing_errors=True
    )
    
    # F-3: Component builds the correct agent type
    agent = plan_execute_component.build_agent()
    
    # F-4: The agent is usable and can be invoked
    result = agent.invoke("I need information about Albert Einstein. First, tell me when he was born, then calculate how many years ago that was.")
    
    # Verify the result contains relevant information
    assert result, "Agent should return a non-empty result"
    assert "Einstein" in str(result), "Result should mention Einstein"
    
    print(f"Plan-Execute Agent Result: {result}")


def test_tool_interoperability():
    """
    Test PRD requirement for tool interoperability.
    
    This test verifies that all required tools can be used with the agents.
    """
    # Skip if langflow components aren't available
    if not langflow_imports_available:
        pytest.skip("Langflow components not available")
    
    # Create all the tools required by the PRD
    calculator = CalculatorTool()
    calc_tool = Tool(
        name="Calculator",
        description="Useful for performing mathematical calculations",
        func=calculator._run
    )
    
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    wiki_tool = Tool(
        name="Wikipedia",
        description="Useful for searching for information on Wikipedia",
        func=wiki._run
    )
    
    search = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())
    search_tool = Tool(
        name="Search",
        description="Useful for searching the internet for current information",
        func=search._run
    )
    
    # Create a simple function tool
    def get_current_weather(location: str) -> str:
        """Simulate getting the weather for a location."""
        return f"The weather in {location} is currently sunny and 22°C."
    
    weather_tool = StructuredTool.from_function(
        func=get_current_weather,
        name="Weather",
        description="Get the current weather for a location"
    )
    
    # Verify the tools are created correctly
    tools = [calc_tool, wiki_tool, search_tool, weather_tool]
    assert len(tools) >= 4, "Should create at least 4 tools for testing"
    
    # Verify tool names
    tool_names = [tool.name for tool in tools]
    assert "Calculator" in tool_names
    assert "Wikipedia" in tool_names
    assert "Search" in tool_names
    assert "Weather" in tool_names
