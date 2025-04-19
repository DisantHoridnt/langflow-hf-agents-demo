"""
Plan-and-Execute Agent Component for Langflow.

This component implements a Plan-and-Execute agent that works with
any LLM, including open-source models hosted on Hugging Face.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from LangChain with fallbacks for different versions
try:
    from langchain.agents import AgentExecutor
    from langchain.agents.react.agent import create_react_agent
    from langchain.agents.plan_and_execute.agent import PlanAndExecute
    from langchain.agents.plan_and_execute.planners.chat_planner import PlanningOutputParser
    from langchain.chains.llm_chain import LLMChain
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.schema.language_model import BaseLanguageModel
    from langchain.tools.base import BaseTool
except ImportError:
    try:
        # Try with newer imports from langchain_* namespace
        from langchain_core.agents import AgentExecutor
        from langchain_community.agents.react.agent import create_react_agent
        from langchain_community.agents.plan_and_execute.agent import PlanAndExecute
        from langchain_community.agents.plan_and_execute.planners.chat_planner import PlanningOutputParser
        from langchain_core.chains import LLMChain
        from langchain_core.memory import BaseChatMemory
        from langchain_core.language_models import BaseLanguageModel
        from langchain_core.tools import BaseTool
    except ImportError:
        logger.warning("Could not import LangChain components")

# Import from our adapter layer instead of directly from Langflow
from .adapters import LCToolsAgentComponent, LanguageModel, Tool, BaseMemory
from langchain_core.runnables import Runnable
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)

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
            name="max_subtask_iterations",
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
    
    def create_planner(self):
        """Create the planner component for the Plan-and-Execute agent.
        
        This creates a planner that breaks down the task into steps.
        """
        if not isinstance(self.llm, BaseLanguageModel):
            raise ValueError(f"Expected llm to be a BaseLanguageModel, got {type(self.llm)}")
        
        # Create the planner prompt template
        planner_prompt = PromptTemplate.from_template(
            "You are a planner that creates a plan to solve a task.\n"
            "{planner_prompt}\n\n"
            "User's request: {input}\n\n"
            "Please create a plan with clear steps. For each step, explain briefly what needs to be done."
        )
        
        # Create the planner chain
        return LLMChain(
            llm=self.llm,
            prompt=planner_prompt,
        )
    
    def create_executor(self):
        """Create the executor component for the Plan-and-Execute agent.
        
        This creates an executor that carries out each step of the plan using available tools.
        """
        if not isinstance(self.llm, BaseLanguageModel):
            raise ValueError(f"Expected llm to be a BaseLanguageModel, got {type(self.llm)}")
        
        if not self.tools:
            raise ValueError("Tools are required for the executor")
        
        # Create the executor prompt
        executor_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "I need to execute this step of the plan: {input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the executor agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=executor_prompt,
        )
        
        # Get callbacks if available
        callbacks: List[BaseCallbackHandler] = []
        if hasattr(self, "get_langchain_callbacks"):
            callbacks = self.get_langchain_callbacks()
        
        # Create the executor
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=self.max_subtask_iterations,
            handle_parsing_errors=self.handle_parsing_errors,
            callbacks=callbacks,
        )
    
    def create_agent_runnable(self) -> Runnable:
        """Create the Plan-and-Execute agent runnable.
        
        This implements the two-stage approach:
        1. Planning: Break down the task into steps
        2. Execution: Carry out each step using available tools
        """
        planner = self.create_planner()
        executor = self.create_executor()
        
        # Get callbacks if available
        callbacks: List[BaseCallbackHandler] = []
        if hasattr(self, "get_langchain_callbacks"):
            callbacks = self.get_langchain_callbacks()
        
        # Create the Plan-and-Execute agent
        return PlanAndExecute(
            planner=planner,
            executor=executor,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            callbacks=callbacks,
        )
    
    def build_agent(self) -> AgentExecutor:
        """Build the Plan-and-Execute agent.
        
        This method validates the tools, creates the planner and executor,
        and returns the complete Plan-and-Execute agent.
        """
        self.validate_tool_names()
        
        # Create the planner and executor
        planner = self.create_planner()
        executor = self.create_executor()
        
        # Get callbacks if available
        callbacks: List[BaseCallbackHandler] = []
        if hasattr(self, "get_langchain_callbacks"):
            callbacks = self.get_langchain_callbacks()
        
        # Create the Plan-and-Execute agent
        agent = PlanAndExecute(
            planner=planner,
            executor=executor,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            callbacks=callbacks,
        )
        
        return agent
