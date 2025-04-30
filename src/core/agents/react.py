"""
Standalone ReAct Agent implementation.

This module provides a ReAct (Reasoning and Acting) agent that works with
any LLM, including open-source models hosted on Hugging Face, without
requiring the Langflow framework.
"""

from typing import Any, Dict, List, Optional

# Updated imports for modern agent creation
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms.base import BaseLLM as BaseLanguageModel
from langchain.tools.base import BaseTool


class StandaloneReActAgent:
    """Standalone implementation of the ReAct Agent.
    
    This agent uses the ReAct (Reasoning and Acting) approach to solve problems
    with any LLM, including open-source models hosted on Hugging Face.
    Uses modern LangChain agent creation functions.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        system_prompt: str = None, # System prompt is now part of the hub prompt
        verbose: bool = True,
        max_iterations: int = 10,
        handle_parsing_errors: bool = True,
        callbacks: List[BaseCallbackHandler] = None,
    ):
        """Initialize the ReAct Agent.
        
        Args:
            llm: The language model to use
            tools: List of tools available to the agent
            system_prompt: (Optional) Base system instructions (less critical now)
            verbose: Whether to print detailed logs
            max_iterations: Maximum number of steps the agent can take
            handle_parsing_errors: Whether to try recovering from parsing errors
            callbacks: Optional callbacks for tracking agent execution
        """
        self.llm = llm
        self.tools = tools
        # System prompt is handled by the hub prompt, but store if provided
        self.system_prompt_override = system_prompt 
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.handle_parsing_errors = handle_parsing_errors
        self.callbacks = callbacks or []
        # Defer executor creation until needed or called explicitly
        # self.agent_executor = self._build_agent_executor()
    
    def _build_agent_executor(self) -> AgentExecutor:
        """Build the AgentExecutor for the ReAct agent using create_react_agent."""
        # Validate tool names to avoid conflicts
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Tool names must be unique")
        
        # Pull the standard ReAct prompt template from Langchain Hub
        # You could potentially customize this prompt further if needed
        prompt = hub.pull("hwchase17/react")
        
        # If a system_prompt was provided, we might want to inject it, 
        # but the hub prompt is usually self-contained. We'll stick to the hub prompt for now.

        # Create the agent using the newer create_react_agent function
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create the AgentExecutor directly with the agent runnable
        executor_kwargs = {
            "agent": agent,
            "tools": self.tools,
            "handle_parsing_errors": self.handle_parsing_errors,
            "max_iterations": self.max_iterations,
            "verbose": self.verbose,
            "callbacks": self.callbacks,
        }
        
        # AgentExecutor initialization is direct now
        return AgentExecutor(**executor_kwargs)
    
    def run(self, input_text: str) -> str:
        """Run the agent on the given input text.
        
        Args:
            input_text: The user's request
            
        Returns:
            The agent's response
        """
        # Build the executor if it doesn't exist
        if not hasattr(self, 'agent_executor'):
             self.agent_executor = self._build_agent_executor()
             
        # Use invoke for LCEL runnables, passing input as dict
        response = self.agent_executor.invoke({"input": input_text})
        # The output is usually in a key like 'output'
        return response.get('output', str(response))
    
    def __call__(self, input_text: str) -> str:
        """Allow the agent to be called directly."""
        return self.run(input_text)
