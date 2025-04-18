"""
Plan-and-Execute Agent Component for Langflow.

This component implements a Plan-and-Execute agent that works with
any LLM, including open-source models hosted on Hugging Face.
"""

from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)

from langflow.base.agents.agent import LCToolsAgentComponent
from langflow.field_typing import LanguageModel, Tool, BaseMemory
from langflow.io import BoolInput, IntInput, MultilineInput


class PlanExecuteAgentComponent(LCToolsAgentComponent):
    """Plan-and-Execute Agent Component for Langflow.
    
    This agent uses a two-stage approach: first planning steps, then executing them.
    Works with any LLM, including open-source models hosted on Hugging Face.
    """

    display_name: str = "Plan-Execute Agent"
    description: str = (
        "Agent using the Plan-and-Execute approach that works with any LLM, "
        "including open-source models from Hugging Face."
    )
    icon = "🗺️"
    beta = False
    name = "PlanExecuteAgentComponent"
    group = "Agents"

    # Define additional inputs specific to the Plan-and-Execute agent
    inputs = [
        MultilineInput(
            name="system_prompt",
            display_name="System Prompt",
            info="System instructions to guide the agent's behavior.",
            value=(
                "You are a helpful assistant that first plans what to do, then executes the plan step by step. "
                "You have access to tools that can help you complete tasks. "
                "Think carefully about the best way to accomplish the user's request."
            ),
            advanced=False,
        ),
        MultilineInput(
            name="planner_prompt",
            display_name="Planner Prompt",
            info="Instructions specifically for the planning phase.",
            value=(
                "Let's break down this task into a specific plan with clear steps. "
                "What is the best approach to accomplish this goal? "
                "Create a plan with no more than 5 steps."
            ),
            advanced=True,
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
        IntInput(
            name="max_execution_iterations",
            display_name="Max Execution Iterations",
            info="Maximum iterations for each execution step.",
            value=5,
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
    
    def set(self, **kwargs: Any) -> "PlanExecuteAgentComponent":
        """Set attributes of the component."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
    
    def create_agent_runnable(self) -> Runnable:
        """Create the Plan-and-Execute agent runnable.
        
        This implements the two-stage approach:
        1. Planning: Break down the task into steps
        2. Execution: Carry out each step using available tools
        """
        if not isinstance(self.llm, BaseLanguageModel):
            raise ValueError(f"Expected llm to be a BaseLanguageModel, got {type(self.llm)}")
        
        if not self.tools:
            raise ValueError("Tools are required for the Plan-and-Execute agent")
        
        # Create the planner
        planner_prompt = PromptTemplate.from_template(
            "You are a planner that creates a plan to solve a task.\n"
            "{planner_prompt}\n\n"
            "User's request: {input}\n\n"
            "Please create a plan with clear steps. For each step, explain briefly what needs to be done."
        )
        
        planner = load_chat_planner(
            llm=self.llm,
            prompt=planner_prompt,
            planner_prompt=self.planner_prompt
        )
        
        # Create the executor
        executor_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "I need to execute this step of the plan: {input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Get callbacks if available
        callbacks: List[BaseCallbackHandler] = []
        if hasattr(self, "get_langchain_callbacks"):
            callbacks = self.get_langchain_callbacks()
        
        executor = load_agent_executor(
            llm=self.llm,
            tools=self.tools,
            prompt=executor_prompt,
            verbose=self.verbose,
            max_iterations=self.max_execution_iterations,
            handle_parsing_errors=self.handle_parsing_errors,
            callbacks=callbacks,
        )
        
        # Create the Plan-and-Execute agent
        return PlanAndExecute(
            planner=planner,
            executor=executor,
            verbose=self.verbose,
            callbacks=callbacks,
        )
    
    def build_agent(self) -> AgentExecutor:
        """Build the Plan-and-Execute agent.
        
        The PlanAndExecute class from LangChain already provides the executor functionality,
        so we return it directly.
        """
        self.validate_tool_names()
        agent = self.create_agent_runnable()
        
        # Prepare kwargs for the AgentExecutor
        executor_kwargs = {
            "agent": agent,
            "tools": self.tools,
            "verbose": self.verbose,
            "max_iterations": self.max_iterations,
            "handle_parsing_errors": self.handle_parsing_errors,
        }
        
        # Add memory if provided (for future v2 implementation)
        if hasattr(self, "memory") and self.memory is not None:
            executor_kwargs["memory"] = self.memory
        
        # Since PlanAndExecute already has the execution logic, we return it
        # wrapped in an AgentExecutor for Langflow compatibility
        executor = AgentExecutor.from_agent_and_tools(**executor_kwargs)
        
        return executor
