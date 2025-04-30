"""Custom ReAct agent component for Langflow."""

from typing import List, Optional, Dict, Any
from langflow import CustomComponent
from langflow.field_typing import BaseLanguageModel, BaseTool
import logging

# Import our ReAct agent implementation
from src.core.agents.react import StandaloneReActAgent

# Configure logging
logger = logging.getLogger(__name__)

class ReactAgentComponent(CustomComponent):
    """Langflow component that implements a ReAct agent compatible with any LLM."""
    
    display_name = "ReAct Agent"
    description = "Agent that uses reasoning and acting to solve tasks with any LLM model."
    icon = "🧠"  # You can use an emoji as the icon
    category = "Agents"
    beta = True
    
    def build_config(self) -> Dict[str, Any]:
        """Define the configuration options for the component."""
        return {
            "llm": {
                "display_name": "Language Model",
                "info": "The language model to use for the agent.",
                "required": True,
            },
            "tools": {
                "display_name": "Tools",
                "info": "The tools the agent can use.",
                "required": True,
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
        tools: List[BaseTool],
        input_value: str,
        max_iterations: int = 10,
        handle_parsing_errors: bool = True,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Build and run the ReAct agent."""
        logger.info(f"Building ReAct agent with {len(tools)} tools")
        
        # Create the agent
        try:
            agent = StandaloneReActAgent(
                llm=llm,
                tools=tools,
                verbose=True,
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
            logger.error(f"Error running ReAct agent: {str(e)}", exc_info=True)
            return f"Error running agent: {str(e)}"
