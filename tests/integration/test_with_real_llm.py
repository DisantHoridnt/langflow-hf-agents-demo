"""Integration tests with real LLMs.

This file contains tests that use real LLMs from Hugging Face.
These tests are marked with pytest.mark.langflow and will be skipped
unless explicitly requested with pytest -m langflow.
"""

import os
import pytest
from dotenv import load_dotenv

from langchain.llms import HuggingFaceHub
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

from src.core.agents import StandaloneReActAgent


# Mark this test module as requiring Langflow and being slow
pytestmark = [pytest.mark.langflow, pytest.mark.slow]


@pytest.fixture(scope="module")
def huggingface_llm():
    """Create a HuggingFace LLM for testing."""
    # Load API token from .env file
    load_dotenv()
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    # Skip tests if no API token is available
    if not api_token:
        pytest.skip("No HUGGINGFACEHUB_API_TOKEN found in .env file")
    
    # Initialize the LLM
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.7, "max_length": 512}
    )
    
    return llm


@pytest.fixture(scope="module")
def wikipedia_tool():
    """Create a Wikipedia tool for testing."""
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    wiki_tool.name = "Lookup"
    return wiki_tool


@pytest.mark.skip(reason="Expensive API call, only run manually")
def test_react_agent_with_real_llm(huggingface_llm, wikipedia_tool):
    """Test the ReAct agent with a real LLM and Wikipedia tool."""
    # Create the agent
    agent = StandaloneReActAgent(
        llm=huggingface_llm,
        tools=[wikipedia_tool],
        max_iterations=3
    )
    
    # Run a simple query
    result = agent.run("What is the capital of France?")
    
    # Basic validation of the result
    assert result, "Result should not be empty"
    assert "Paris" in result, "Result should contain 'Paris'"
