"""Custom ReAct Agent component for Langflow.

This component implements a ReAct (Reasoning + Acting) agent that can work with
any LLM, including open-source models without native tool binding capabilities.
"""

from typing import List, Optional
from langflow import CustomComponent
from langflow.field_typing import LanguageModel, Tool, BaseLanguageModel
import logging

# Import our custom agent implementation
from src.core.agents.react import StandaloneReActAgent

# Configure logging
logger = logging.getLogger(__name__)

class ReActAgent(CustomComponent):
    """ReAct Agent compatible with any LLM."""
    
    display_name = "ReAct Agent"
    description = "Agent that can use any LLM to reason and act with tools."
    icon = "🧠"
    category = "Agents"
    
    def build_config(self):
        """Define the configuration for the component."""
        return {
            "llm": {
                "display_name": "Language Model",
                "info": "The language model to use for the agent.",
                "required": True,
                "input_types": ["BaseLanguageModel", "LanguageModel", "ChatModel"],
            },
            "tools": {
                "display_name": "Tools",
                "info": "The tools the agent can use.",
                "required": True,
                "is_list": True,
                "input_types": ["Tool"],
            },
            "input_value": {
                "display_name": "Input",
                "info": "The input query or task for the agent.",
                "required": True,
            },
            "max_iterations": {
                "display_name": "Max Iterations",
                "info": "Maximum number of reasoning steps.",
                "advanced": True,
                "value": 10,
            },
            "verbose": {
                "display_name": "Verbose",
                "info": "Whether to print detailed logs during execution.",
                "advanced": True,
                "value": True,
            },
            "handle_parsing_errors": {
                "display_name": "Handle Parsing Errors",
                "info": "Try to recover from LLM output parsing errors.",
                "advanced": True,
                "value": True,
            },
            "system_prompt": {
                "display_name": "Custom System Prompt",
                "info": "Optional: Override the default system prompt.",
                "advanced": True,
                "required": False,
            },
        }
    
    def build(
        self,
        llm: BaseLanguageModel,
        tools: List[Tool],
        input_value: str,
        max_iterations: int = 10,
        verbose: bool = True,
        handle_parsing_errors: bool = True,
        system_prompt: Optional[str] = None,
        **kwargs,  # Accept additional keyword arguments
    ) -> str:
        """Build and run the ReAct agent."""
        logger.info(f"Building ReAct agent with {len(tools)} tools")
        
        try:
            # Create the agent
            agent = StandaloneReActAgent(
                llm=llm,
                tools=tools,
                verbose=verbose,
                max_iterations=max_iterations,
                handle_parsing_errors=handle_parsing_errors,
                system_prompt=system_prompt,
            )
            
            # Run the agent
            logger.info(f"Running ReAct agent with input: {input_value[:50]}...")
            result = agent.run(input_value)
            logger.info(f"Agent execution completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Error running ReAct agent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
