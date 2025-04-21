"""
Google Search Example for Langflow Agents

This example demonstrates how to use the Google Search tool with Langflow agents
in a production environment.
"""

import os
import logging
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from LangChain namespaces
from langchain_core.tools import BaseTool
from langchain_community.tools.calculator.tool import CalculatorTool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.chat_models import ChatOpenAI

# Import Langflow components
from src.integrations.langflow import ReActAgentComponent
from src.integrations.langflow.tools import get_google_search_tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run the example."""
    # Check if API keys are set
    if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("GOOGLE_CSE_ID"):
        logger.error(
            "Please set the GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables."
            "\nYou can get these from the Google Cloud Console and Programmable Search Engine."
        )
        return
    
    # Set up LLM using Hugging Face
    from langchain_community.llms import HuggingFaceEndpoint
    
    # Use Hugging Face model with API key from environment
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.1,
        max_length=1024,
    )
    
    # Set up tools
    tools = [
        # Google Search tool (more reliable than DuckDuckGo)
        get_google_search_tool(),
        
        # Standard calculator tool
        CalculatorTool(),
        
        # Wikipedia tool for reference information
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    ]
    
    # Create the ReAct agent with Google Search
    react_component = ReActAgentComponent()
    react_component.set(
        llm=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful AI assistant that can use tools to answer the user's question. "
            "For searches about current information, use the google_search tool. "
            "For calculations, use the calculator tool. "
            "For reference information, use the wikipedia tool."
        ),
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )
    
    # Build the agent
    agent = react_component.build_agent()
    
    # Run the agent
    result = agent.invoke({
        "input": "What was the highest grossing movie of 2023 and how much did it make?"
    })
    
    # Print the output
    print("\n======== AGENT RESPONSE ========")
    print(result["output"])
    
    # Print the reasoning steps if available
    if "intermediate_steps" in result:
        print("\n======== AGENT REASONING ========")
        for step in result["intermediate_steps"]:
            print(f"Tool: {step[0].tool}")
            print(f"Input: {step[0].tool_input}")
            print(f"Output: {step[1]}")
            print()


if __name__ == "__main__":
    main()
