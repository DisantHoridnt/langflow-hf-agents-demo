"""
Standalone ReAct Agent implementation.

This module provides a ReAct (Reasoning and Acting) agent that works with
any LLM, including open-source models hosted on Hugging Face, without
requiring the Langflow framework.
"""

from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor
from langchain.agents.react.base import ReActDocstoreAgent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms.base import BaseLLM as BaseLanguageModel
from langchain.tools.base import BaseTool


class StandaloneReActAgent:
    """Standalone implementation of the ReAct Agent.
    
    This agent uses the ReAct (Reasoning and Acting) approach to solve problems
    with any LLM, including open-source models hosted on Hugging Face.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        system_prompt: str = None,
        verbose: bool = True,
        max_iterations: int = 10,
        handle_parsing_errors: bool = True,
        callbacks: List[BaseCallbackHandler] = None,
    ):
        """Initialize the ReAct Agent.
        
        Args:
            llm: The language model to use
            tools: List of tools available to the agent
            system_prompt: System instructions for the agent
            verbose: Whether to print detailed logs
            max_iterations: Maximum number of steps the agent can take
            handle_parsing_errors: Whether to try recovering from parsing errors
            callbacks: Optional callbacks for tracking agent execution
        """
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that has access to the following tools. "
            "You should carefully analyze the user's request and use tools when needed to fulfill it. "
            "For each tool use, carefully review the results before moving on."
        )
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.handle_parsing_errors = handle_parsing_errors
        self.callbacks = callbacks or []
        self.agent_executor = self._build_agent_executor()
    
    def _build_agent_executor(self) -> AgentExecutor:
        """Build the AgentExecutor for the ReAct agent."""
        # Validate tool names to avoid conflicts
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Tool names must be unique")
        
        # Create the ReAct agent using the older API available in langchain 0.0.310
        agent = ReActDocstoreAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            system_message=self.system_prompt,
            verbose=self.verbose
        )
        
        # Create the AgentExecutor with the right configuration
        executor_kwargs = {
            "agent": agent,
            "tools": self.tools,
            "handle_parsing_errors": self.handle_parsing_errors,
            "max_iterations": self.max_iterations,
            "verbose": self.verbose,
            "callbacks": self.callbacks,
        }
        
        return AgentExecutor.from_agent_and_tools(**executor_kwargs)
    
    def run(self, input_text: str) -> str:
        """Run the agent on the given input text.
        
        Args:
            input_text: The user's request
            
        Returns:
            The agent's response
        """
        return self.agent_executor.run(input_text)
    
    def __call__(self, input_text: str) -> str:
        """Allow the agent to be called directly."""
        return self.run(input_text)
