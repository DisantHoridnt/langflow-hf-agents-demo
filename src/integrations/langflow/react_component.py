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
except ImportError:
    try:
        # Try with newer imports
        from langchain_core.memory import BaseChatMemory
        from langchain_core.language_models import BaseLanguageModel
        from langchain_core.runnables import Runnable
        from langchain_core.tools import BaseTool
    except ImportError:
        logger.warning("Could not import LangChain base components")

# Import our custom adapter instead of directly from Langflow
from .adapters import LCToolsAgentComponent, LanguageModel, Tool, BaseMemory

# Import Langflow input types or use our own if not available
try:
    from langflow.io import BoolInput, IntInput, MultilineInput
except ImportError:
    logger.warning("Could not import Langflow input types - using fallbacks")
    
    # Define basic input types if imports fail
    class MultilineInput:
        def __init__(self, *, name, display_name, info, value, advanced=False):
            self.name = name
            self.display_name = display_name
            self.info = info
            self.value = value
            self.advanced = advanced

    class IntInput:
        def __init__(self, *, name, display_name, info, value, advanced=False):
            self.name = name
            self.display_name = display_name
            self.info = info
            self.value = value
            self.advanced = advanced

    class BoolInput:
        def __init__(self, *, name, display_name, info, value, advanced=False):
            self.name = name
            self.display_name = display_name
            self.info = info
            self.value = value
            self.advanced = advanced


class ReActAgentComponent(LCToolsAgentComponent):
    """ReAct Agent Component for Langflow.
    
    This agent uses the ReAct (Reasoning and Acting) approach to solve problems
    with any LLM, including open-source models hosted on Hugging Face.
    """

    display_name: str = "ReAct Agent"
    description: str = (
        "Agent using the ReAct (Reasoning and Acting) approach that works with any LLM, "
        "including open-source models from Hugging Face."
    )
    icon = "🤔"
    beta = False
    name = "ReActAgentComponent"
    group = "Agents"

    # Define additional inputs specific to the ReAct agent
    inputs = [
        MultilineInput(
            name="system_prompt",
            display_name="System Prompt",
            info="System instructions to guide the agent's behavior.",
            value=(
                "You are a helpful assistant that has access to the following tools. "
                "You should carefully analyze the user's request and use tools when needed to fulfill it. "
                "For each tool use, carefully review the results before moving on."
            ),
            advanced=False,
        ),
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
        """Create the ReAct agent runnable using the provided LLM and tools.
        
        This is the core implementation of the ReAct agent pattern.
        """
        if not isinstance(self.llm, BaseLanguageModel):
            raise ValueError(f"Expected llm to be a BaseLanguageModel, got {type(self.llm)}")
        
        if not self.tools:
            raise ValueError("Tools are required for the ReAct agent")
        
        # Create the prompt template with the right structure for ReAct
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the ReAct agent
        return create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
    
    def build_agent(self) -> AgentExecutor:
        """Build and configure the AgentExecutor for the ReAct agent."""
        self.validate_tool_names()
        
        # Get callbacks if available (from Langflow integration)
        callbacks: List[BaseCallbackHandler] = []
        if hasattr(self, "get_langchain_callbacks"):
            callbacks = self.get_langchain_callbacks()
        
        # Create the agent runnable
        agent = self.create_agent_runnable()
        
        # Create the AgentExecutor with the right configuration
        executor_kwargs = {
            "agent": agent,
            "tools": self.tools,
            "handle_parsing_errors": self.handle_parsing_errors,
            "max_iterations": self.max_iterations,
            "verbose": self.verbose,
            "callbacks": callbacks,
        }
        
        # Add memory if provided (for future v2 implementation)
        if hasattr(self, "memory") and self.memory is not None:
            executor_kwargs["memory"] = self.memory
        
        # Create the AgentExecutor with the right configuration
        return AgentExecutor.from_agent_and_tools(**executor_kwargs)
