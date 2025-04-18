"""
Test script for the standalone agent implementations.

This script demonstrates how to use the standalone agents with Hugging Face models.
"""

import os
from dotenv import load_dotenv

from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain.llms import HuggingFaceHub

from standalone_agents import StandaloneReActAgent

# Load environment variables including HUGGINGFACEHUB_API_TOKEN
load_dotenv()


def check_api_key():
    """Check if the Hugging Face API token is set."""
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_key:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN environment variable is not set. "
            "Please set it in the .env file."
        )
    return api_key


def create_llm():
    """Create a Hugging Face LLM."""
    api_key = check_api_key()
    
    # Use Mistral model for testing with chat template
    # For older langchain versions, we need to be more careful with the configuration
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",  # Using a simpler model that works well with the older API
        huggingfacehub_api_token=api_key,
        model_kwargs={
            "temperature": 0.7,
            "max_length": 512
        }
    )
    return llm


def create_tools():
    """Create a set of tools for the agents to use."""
    # Create Wikipedia tool with the name 'Lookup'
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    wikipedia.name = "Lookup"
    wikipedia.description = "Lookup information in Wikipedia"
    
    # Create DuckDuckGo search tool with the name 'Search'
    duckduckgo = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())
    duckduckgo.name = "Search"
    duckduckgo.description = "Search for information on the web"
    
    return [wikipedia, duckduckgo]


def test_react_agent():
    """Test the ReAct agent."""
    print("\n===== Testing ReAct Agent =====")
    
    try:
        llm = create_llm()
        tools = create_tools()
        
        agent = StandaloneReActAgent(
            llm=llm,
            tools=tools,
            verbose=True
        )
        
        question = "What is the capital of France and what is the population?"
        print(f"\nQuestion: {question}")
        
        answer = agent.run(question)
        print(f"\nAnswer: {answer}")
        
        return True
    
    except Exception as e:
        print(f"Error testing ReAct agent: {e}")
        return False


# We've removed the Plan-Execute agent test since it requires experimental modules


if __name__ == "__main__":
    print("Starting tests for standalone agents...")
    
    react_success = test_react_agent()
    
    if react_success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Test failed. Please check error messages above.")
