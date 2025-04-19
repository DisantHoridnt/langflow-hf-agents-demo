"""
Plan-and-Execute Agent Component for Langflow.

This component implements a Plan-and-Execute agent that works with
any LLM, including open-source models hosted on Hugging Face.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Initialize names to None
PromptTemplate = None
LLMChain = None
AgentExecutor = None
create_react_agent = None
PlanAndExecute = None
PlanningOutputParser = None
BaseChatMemory = None
BaseLanguageModel = None
BaseTool = None

# Import from LangChain with fallbacks for different versions
logger.debug("Attempting LangChain imports (older style)...")
try:
    from langchain.agents import AgentExecutor as LangchainAgentExecutor
    logger.debug("Successfully imported AgentExecutor (older)")
    from langchain.agents.react.agent import create_react_agent as LangchainCreateReactAgent
    logger.debug("Successfully imported create_react_agent (older)")
    from langchain.agents.plan_and_execute.agent import PlanAndExecute as LangchainPlanAndExecute
    logger.debug("Successfully imported PlanAndExecute (older)")
    from langchain.agents.plan_and_execute.planners.chat_planner import PlanningOutputParser as LangchainPlanningOutputParser
    logger.debug("Successfully imported PlanningOutputParser (older)")
    from langchain.chains.llm_chain import LLMChain as LangchainLLMChain
    logger.debug("Successfully imported LLMChain (older)")
    from langchain.memory.chat_memory import BaseChatMemory as LangchainBaseChatMemory
    logger.debug("Successfully imported BaseChatMemory (older)")
    from langchain.schema.language_model import BaseLanguageModel as LangchainBaseLanguageModel
    logger.debug("Successfully imported BaseLanguageModel (older)")
    from langchain.tools.base import BaseTool as LangchainBaseTool
    logger.debug("Successfully imported BaseTool (older)")
    from langchain.prompts import PromptTemplate as LangchainPromptTemplate
    logger.debug("Successfully imported PromptTemplate (older)")

    AgentExecutor = LangchainAgentExecutor
    create_react_agent = LangchainCreateReactAgent
    PlanAndExecute = LangchainPlanAndExecute
    PlanningOutputParser = LangchainPlanningOutputParser
    LLMChain = LangchainLLMChain
    BaseChatMemory = LangchainBaseChatMemory
    BaseLanguageModel = LangchainBaseLanguageModel
    BaseTool = LangchainBaseTool
    PromptTemplate = LangchainPromptTemplate
    logger.debug("Assigned older LangChain imports successfully.")

except ImportError as e_old:
    logger.warning(f"Older LangChain imports failed: {e_old}. Attempting newer style imports...")
    try:
        # Try with newer imports from langchain_* namespace
        logger.debug("Attempting LangChain imports (newer style)...")
        from langchain_core.agents import AgentExecutor as CoreAgentExecutor
        logger.debug("Successfully imported AgentExecutor (newer)")
        AgentExecutor = CoreAgentExecutor

        from langchain_community.agents.react.agent import create_react_agent as CommunityCreateReactAgent
        logger.debug("Successfully imported create_react_agent (newer)")
        create_react_agent = CommunityCreateReactAgent

        # PlanAndExecute might still be experimental or have different paths
        # from langchain_community.agents.plan_and_execute.agent import PlanAndExecute as CommunityPlanAndExecute
        # from langchain_community.agents.plan_and_execute.planners.chat_planner import PlanningOutputParser as CommunityPlanningOutputParser
        
        from langchain.chains import LLMChain as CoreLLMChain # Langchain 0.1.x path
        logger.debug("Successfully imported LLMChain (newer style path: langchain.chains)")
        LLMChain = CoreLLMChain

        from langchain_core.memory import BaseChatMemory as CoreBaseChatMemory
        logger.debug("Successfully imported BaseChatMemory (newer)")
        BaseChatMemory = CoreBaseChatMemory

        from langchain_core.language_models import BaseLanguageModel as CoreBaseLanguageModel
        logger.debug("Successfully imported BaseLanguageModel (newer)")
        BaseLanguageModel = CoreBaseLanguageModel

        from langchain_core.tools import BaseTool as CoreBaseTool
        logger.debug("Successfully imported BaseTool (newer)")
        BaseTool = CoreBaseTool

        from langchain_core.prompts import PromptTemplate as CorePromptTemplate
        logger.debug("Successfully imported PromptTemplate (newer)")
        PromptTemplate = CorePromptTemplate # Assign immediately
        logger.debug(f"Assigned PromptTemplate (newer). Current value type: {type(PromptTemplate)}") # Log assignment success and type

        logger.debug("Assigned newer LangChain imports successfully.")

    except ImportError as e_new:
        logger.error(f"Newer LangChain imports also failed: {e_new}")
        # Set required components to None if secondary import fails
        PromptTemplate = None
        LLMChain = None
        BaseLanguageModel = None # Ensure this is also reset if newer imports fail

# Check if essential imports were successful before proceeding
logger.debug(f"Checking essential components: PromptTemplate={PromptTemplate is not None}, LLMChain={LLMChain is not None}, BaseLanguageModel={BaseLanguageModel is not None}")
if PromptTemplate is None or LLMChain is None or BaseLanguageModel is None:
    logger.error(
        "Could not import essential LangChain components (PromptTemplate, LLMChain, BaseLanguageModel). "
        "Please ensure LangChain is installed correctly."
    )

# Import from our adapter layer instead of directly from Langflow
from .adapters import LCToolsAgentComponent, LanguageModel, Tool, BaseMemory, BoolInput, IntInput, MultilineInput
from langchain_core.runnables import Runnable
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)

class PlanExecuteAgentComponent(LCToolsAgentComponent):
    """Plan-and-Execute Agent Component for Langflow.
    
    This agent uses a two-stage approach: first planning steps, then executing them.
    Works with any LLM, including open-source models hosted on Hugging Face.
    """

    display_name: str = "Plan-and-Execute Agent"
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
        
        This creates a planner that outlines the steps needed to achieve the goal.
        """
        if self.llm is None:
            raise ValueError("LLM is required for the planner")
        
        # Check for necessary attributes instead of using isinstance check
        if not (hasattr(self.llm, "invoke") or hasattr(self.llm, "generate") or hasattr(self.llm, "__call__")):
            raise ValueError(f"LLM needs to have invoke, generate, or be callable. Got {type(self.llm)}")
        
        # Check if system prompt is provided
        if not self.system_prompt:
            raise ValueError("System Prompt is required for the planner.")

        # Local import of necessary components to avoid depending on global imports
        try:
            # First try langchain_core (newer versions)
            logger.debug("Attempting to import PromptTemplate from langchain_core.prompts")
            from langchain_core.prompts import PromptTemplate as LocalPromptTemplate
            logger.debug("Successfully imported PromptTemplate from langchain_core.prompts")
        except ImportError:
            try:
                # Then try langchain (older versions)
                logger.debug("Attempting to import PromptTemplate from langchain.prompts")
                from langchain.prompts import PromptTemplate as LocalPromptTemplate
                logger.debug("Successfully imported PromptTemplate from langchain.prompts")
            except ImportError:
                raise ImportError("Could not import PromptTemplate from either langchain_core.prompts or langchain.prompts")

        # Construct the planner prompt template using local import
        logger.debug(f"Using locally imported PromptTemplate")
        planner_prompt = LocalPromptTemplate.from_template(
            template=self.planner_prompt, # Use the input field value
        )
        
        # Modern approach: Use RunnableSequence (prompt | llm) pattern instead of LLMChain
        try:
            # Import RunnablePassthrough if needed
            from langchain_core.runnables import RunnablePassthrough
            logger.debug("Using modern RunnableSequence pattern instead of deprecated LLMChain")
            
            # Create a chain using the | operator (RunnableSequence)
            planner_chain = planner_prompt | self.llm
            
            # For logging purposes only
            logger.debug("Successfully created planner: RunnableSequence")
            return planner_chain
            
        except ImportError as e:
            logger.warning(f"Could not use modern RunnableSequence pattern: {e}. Falling back to LLMChain.")
            # Fallback to LLMChain if needed
            try:
                # First try newer imports
                from langchain.chains import LLMChain as LocalLLMChain
                logger.debug("Falling back to LLMChain from langchain.chains")
            except ImportError:
                try:
                    # Then try older imports
                    from langchain.chains.llm_chain import LLMChain as LocalLLMChain
                    logger.debug("Falling back to LLMChain from langchain.chains.llm_chain")
                except ImportError:
                    raise ImportError("Could not import LLMChain from either langchain.chains or langchain.chains.llm_chain")

            # Create the planner chain with locally imported class
            return LocalLLMChain(
                llm=self.llm,
                prompt=planner_prompt,
            )
    
    def create_executor(self):
        """Create the executor component for the Plan-and-Execute agent.
        
        This creates an executor that carries out each step of the plan using available tools.
        """
        if self.llm is None:
            raise ValueError("LLM is required for the planner")
        
        # Check for necessary attributes instead of using isinstance check
        if not (hasattr(self.llm, "invoke") or hasattr(self.llm, "generate") or hasattr(self.llm, "__call__")):
            raise ValueError(f"LLM needs to have invoke, generate, or be callable. Got {type(self.llm)}")
        
        if not self.tools:
            raise ValueError("Tools are required for the executor")
        
        # Create the executor prompt
        # We need ChatPromptTemplate and MessagesPlaceholder here as well
        # Let's import them dynamically within the method for now to avoid NameErrors
        try:
            from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        except ImportError:
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            
        # Create a formatted string with tool descriptions for the prompt
        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        # Create a dictionary that we can use to format the message template
        tool_names = [tool.name for tool in self.tools]
        
        # Create the executor prompt including all required tool information
        executor_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + "\n\nYou have access to the following tools:\n" + tool_strings),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "I need to execute this step of the plan: {input}\n\nThe available tools are: {tool_names}\n\nTools: {tools}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Partial format the template to supply tools and tool_names which are fixed for this agent
        executor_prompt = executor_prompt.partial(tools=str(self.tools), tool_names=", ".join(tool_names))
        
        # Local import for create_react_agent
        try:
            # Try newer imports first
            from langchain_community.agents.react.agent import create_react_agent as local_create_react_agent
            logger.debug("Successfully imported create_react_agent from langchain_community")
        except ImportError:
            try:
                # Try older imports
                from langchain.agents.react.agent import create_react_agent as local_create_react_agent
                logger.debug("Successfully imported create_react_agent from langchain")
            except ImportError:
                raise ImportError("Could not import create_react_agent from either langchain_community or langchain")
        
        # Create the executor agent with locally imported function
        agent = local_create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=executor_prompt,
        )
        
        # Get callbacks if available
        callbacks: List[BaseCallbackHandler] = []
        if hasattr(self, "get_langchain_callbacks"):
            callbacks = self.get_langchain_callbacks()
        
        # Local import for AgentExecutor
        try:
            # Try newer imports first
            from langchain_core.agents import AgentExecutor as LocalAgentExecutor
            logger.debug("Successfully imported AgentExecutor from langchain_core")
        except ImportError:
            try:
                # Try older imports
                from langchain.agents import AgentExecutor as LocalAgentExecutor
                logger.debug("Successfully imported AgentExecutor from langchain")
            except ImportError:
                raise ImportError("Could not import AgentExecutor from either langchain_core or langchain")
        
        # Create the executor with locally imported class
        return LocalAgentExecutor.from_agent_and_tools(
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
        
        # Local import for PlanAndExecute and related components
        try:
            # Try importing directly from langchain_experimental
            from langchain_experimental.plan_and_execute import PlanAndExecute as LocalPlanAndExecute
            logger.info("Successfully imported PlanAndExecute from langchain_experimental")
        except ImportError:
            try:
                # Try older import path
                from langchain.agents.plan_and_execute.agent import PlanAndExecute as LocalPlanAndExecute
                logger.info("Successfully imported PlanAndExecute from langchain")
            except ImportError:
                raise ImportError("Could not import PlanAndExecute from either langchain_experimental or langchain")
        
        # Create planner and executor with the methods we've fixed
        planner = self.create_planner()
        executor = self.create_executor()
        
        # Create the Plan-and-Execute agent with locally imported class
        return LocalPlanAndExecute(
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
        
        # Local import for PlanAndExecute class to avoid global import issues
        try:
            # Try importing directly from langchain_experimental
            from langchain_experimental.plan_and_execute import PlanAndExecute as LocalPlanAndExecute
            logger.info("Successfully imported PlanAndExecute from langchain_experimental")
        except ImportError:
            try:
                # Try older import path
                from langchain.agents.plan_and_execute.agent import PlanAndExecute as LocalPlanAndExecute
                logger.info("Successfully imported PlanAndExecute from langchain")
            except ImportError:
                raise ImportError("Could not import PlanAndExecute from either langchain_experimental or langchain")
        
        # Create the planner and executor
        planner = self.create_planner()
        executor = self.create_executor()
        
        # Get callbacks if available
        callbacks: List[BaseCallbackHandler] = []
        if hasattr(self, "get_langchain_callbacks"):
            callbacks = self.get_langchain_callbacks()
        
        # Create the Plan-and-Execute agent using the locally imported class
        agent = LocalPlanAndExecute(
            planner=planner,
            executor=executor,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            callbacks=callbacks,
        )
        
        return agent
