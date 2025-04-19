"""
ReAct Agent Component for Langflow.

This component implements a ReAct (Reasoning and Acting) agent that works with
any LLM, including open-source models hosted on Hugging Face.
"""

import os
import logging
from typing import Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from LangChain - using try/except for flexibility
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.agents.format_scratchpad.openai_tools import (
        format_to_openai_tool_messages,
    )
    from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
except ImportError:
    try:
        # Try with newer imports
        from langchain_core.agents import AgentExecutor
        from langchain_community.agents.react.agent import create_react_agent
        from langchain_core.agents.format_scratchpad.openai_tools import (
            format_to_openai_tool_messages,
        )
        from langchain_core.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
    except ImportError:
        logger.warning("Could not import LangChain agent components")

try:
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.schema.language_model import BaseLanguageModel
    from langchain.schema.runnable import Runnable
    from langchain.tools.base import BaseTool
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:
    try:
        # Try with newer imports
        from langchain_core.memory import BaseChatMemory
        from langchain_core.language_models import BaseLanguageModel
        from langchain_core.runnables import Runnable
        from langchain_core.tools import BaseTool
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    except ImportError:
        logger.warning("Could not import LangChain base components")

# Import our custom adapter instead of directly from Langflow
from .adapters import LCToolsAgentComponent, LanguageModel, Tool, BaseMemory, BoolInput, IntInput, MultilineInput
from langchain import hub

# Define a constant for the default system prompt
DEFAULT_SYSTEM_PROMPT = """Assistant is a large language model trained by Google."""

class ReActAgentComponent(LCToolsAgentComponent):
    """ReAct Agent Component for Langflow.
    
    This agent uses the ReAct (Reasoning and Acting) approach to solve problems
    with any LLM, including open-source models hosted on Hugging Face.
    """

    display_name: str = "ReAct Agent"
    description: str = "Construct a ReAct agent from an LLM and tools."
    documentation: str = "https://python.langchain.com/docs/modules/agents/agent_types/react"
    icon = "Langchain"

    # Define additional inputs specific to the ReAct agent
    inputs = [
        BoolInput(
            name="verbose",
            display_name="Verbose",
            info="Print detailed logs of the agent's thought process.",
            value=True,
            advanced=True,
        ),
        IntInput(
            name="max_iterations",
            display_name="Max Iterations",
            info="Maximum number of steps the agent can take before stopping.",
            value=10,
            advanced=True,
        ),
        BoolInput(
            name="handle_parsing_errors",
            display_name="Handle Parsing Errors",
            info="Attempt to recover from agent output parsing errors.",
            value=True,
            advanced=True,
        ),
        ("memory", BaseMemory, None),  # Optional memory input for future v2 implementation
        *LCToolsAgentComponent._base_inputs
    ]
    
    llm: Optional[LanguageModel] = None
    tools: Optional[List[Tool]] = None
    
    def set(self, **kwargs: Any) -> "ReActAgentComponent":
        """Set attributes of the component."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
    
    def create_agent_runnable(self) -> Runnable:
        """Create the ReAct agent runnable."""
        if not self.llm:
            raise ValueError("Language Model (LLM) is not set.")
        if not self.tools:
            logger.warning("No tools provided for the agent.")
            tools_list = []
        else:
            tools_list = self.tools

        # Get the ReAct prompt
        # TODO: Make the prompt customizable via input
        prompt = hub.pull("hwchase17/react")

        if not create_react_agent:
            raise ImportError("create_react_agent function not available from LangChain imports.")
            
        # Construct the ReAct agent
        agent = create_react_agent(self.llm, tools_list, prompt)
        return agent

    def build_agent(self) -> AgentExecutor:
        """Build the ReAct agent executor."""
        if not AgentExecutor:
            raise ImportError("AgentExecutor class not available from LangChain imports.")
            
        agent_runnable = self.create_agent_runnable()

        # Create the AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent_runnable,
            tools=self.tools or [],
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            handle_parsing_errors=self.handle_parsing_errors,
            # memory=self.memory, # TODO: Integrate memory if provided
        )
        return agent_executor
