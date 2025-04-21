"""
ReAct Agent Component for Langflow.

This component implements a ReAct (Reasoning and Acting) agent that works with
any LLM, including open-source models hosted on Hugging Face.
"""

import os
import logging
from typing import Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.WARNING)
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

# Import our adapters
from src.integrations.langflow.adapters import (
    LCToolsAgentComponent,
    BoolInput,
    IntInput,
    MultilineInput,
    BaseMemory,
    Tool,
    LanguageModel
)

# Import Input class from our adapter module to ensure consistent fallback behavior
from .adapters import LCToolsAgentComponent

# Get Input class - this will use our robust import system in adapters.py
try:
    # First try to access Input class directly from the LCToolsAgentComponent
    Input = LCToolsAgentComponent.Input
except AttributeError:
    # If not available, try common Langflow paths
    try:
        from langflow.interface.input_interface import Input
    except ImportError:
        try:
            from langflow.interface.base import Input
        except ImportError:
            try:
                from langflow.custom.customs import Input
            except ImportError:
                # Create our own minimal implementation if nothing works
                class Input:
                    def __init__(self, name, type=None, default=None, display_name=None, info=None, advanced=False):
                        self.name = name
                        self.type = type
                        self.default = default
                        self.display_name = display_name or name
                        self.info = info
                        self.advanced = advanced

from langchain import hub

# Import the scalable tool enhancement system
from .tools.integration import enhance_tools_for_agent

# Define a constant for the default system prompt with enhanced tool usage examples
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that can use tools to answer the user's question.
Use the provided tools to find information and solve problems step by step.

IMPORTANT TOOL USAGE GUIDE:
- Wikipedia: Use for factual knowledge and information about topics. Input only the search term.
  Example: Action: Wikipedia
           Action Input: France

- Calculator: Use ONLY for mathematical calculations with numbers and operators.
  Example: Action: Calculator
           Action Input: 2 + 2
  Note: Only use operators like +, -, *, /, ^, (), not words or sentences.

- Search: Use for current events or broad web searches. Input only the search query.
  Example: Action: Search
           Action Input: latest news about AI

If you receive an error from a tool, analyze what went wrong and try a different approach.
Maintain the correct format at all times: Thought, Action, Action Input, Observation.
"""

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
        # Use proper Input class for memory instead of tuple
        Input(
            name="memory",
            type=BaseMemory,
            default=None,
            display_name="Memory",
            info="Optional memory for conversation history (for future v2 implementation)",
            advanced=True,
        ),
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
            # Use the new scalable tool processing system with proper middleware
            tools_list = enhance_tools_for_agent(self.tools)
            logger.info(f"Enhanced {len(tools_list)} tools with specialized processors and middleware")

        # Create a custom enhanced ReAct prompt with better tool usage examples
        try:
            # Try to get the standard React prompt as a base
            base_prompt = hub.pull("hwchase17/react")
            
            # Enhance with our custom system message
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
            
            # Extract the template messages
            messages = base_prompt.messages
            # Replace the system message with our enhanced version
            for i, message in enumerate(messages):
                if hasattr(message, 'prompt') and hasattr(message.prompt, 'template'):
                    # This is likely the system message
                    messages[i] = SystemMessagePromptTemplate.from_template(DEFAULT_SYSTEM_PROMPT)
                    break
                    
            # Create enhanced prompt
            prompt = ChatPromptTemplate.from_messages(messages)
            logger.info("Successfully enhanced React prompt with custom tool usage examples")
        except Exception as e:
            # Fallback to original prompt if enhancement fails
            logger.warning(f"Failed to enhance prompt: {str(e)}. Using default React prompt.")
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

        # Create a format fixer to help the agent recover when confused
        def format_fixer(error: Exception, observation: str) -> str:
            """Help the agent recover from formatting errors.
            
            This provides a standardized way to get the agent back on track
            if it gets confused about the expected format.
            """
            # Check for format errors
            error_message = str(error)
            
            if "format" in error_message.lower() or "missing 'action'" in error_message.lower():
                return (
                    "I notice you didn't follow the format correctly. Remember to:\n"
                    "1. Use 'Thought:' to express your reasoning\n"
                    "2. Use 'Action:' to specify which tool to use\n"
                    "3. Use 'Action Input:' to provide input to the tool\n"
                    "4. When you're ready to give the final answer, use 'Final Answer:'\n"
                    "\nLet me try again with the proper format."
                )
            
            # For other errors
            return f"There was an error: {error_message}. Let me try a different approach."
            
        # Create the AgentExecutor with explicit input_key and the format fixer
        agent_executor = AgentExecutor(
            agent=agent_runnable,
            tools=self.tools or [],
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            handle_parsing_errors=self.handle_parsing_errors,
            # memory=self.memory, # TODO: Integrate memory if provided
            input_key="input",  # Specify the input key to handle string inputs properly
            handle_tool_error=format_fixer  # Add our custom format fixer
        )
        return agent_executor
