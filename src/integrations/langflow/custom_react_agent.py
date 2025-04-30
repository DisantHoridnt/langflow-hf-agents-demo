# src/integrations/langflow/custom_react_agent.py

from typing import List, Optional
from langflow import CustomComponent
from langflow.field_typing import LanguageModel, Tool, BaseLanguageModel, BaseTool
from src.core.agents.react import StandaloneReActAgent # Import our agent logic
from langchain.callbacks.base import BaseCallbackHandler

class ReActAgentComponent(CustomComponent):
    display_name = "Custom ReAct Agent"
    description = "Runs a ReAct agent using the provided LLM and tools, compatible with models lacking native tool binding."
    beta: bool = True # Mark as beta as it's custom

    def build_config(self):
        return {
            "llm": {"display_name": "Language Model", "required": True, "info": "The LLM to use for the agent."},
            "tools": {"display_name": "Tools", "required": True, "info": "The list of tools the agent can use."},
            "input_value": {"display_name": "Input", "required": True, "info": "The input query or task for the agent."},
            "max_iterations": {"display_name": "Max Iterations", "value": 10, "advanced": True, "info": "Maximum reasoning steps."},
            "handle_parsing_errors": {"display_name": "Handle Parsing Errors", "value": True, "advanced": True, "info": "Attempt to recover from LLM output parsing errors."},
            "system_prompt": {"display_name": "System Prompt (Optional)", "required": False, "advanced": True, "info": "Overrides the default ReAct system prompt (use with care)."},
            # We might need to handle callbacks if Langflow integrates them, but start simple
        }

    def build(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        input_value: str,
        max_iterations: int = 10,
        handle_parsing_errors: bool = True,
        system_prompt: Optional[str] = None,
        # callbacks: List[BaseCallbackHandler] = None, # Handle later if needed
    ) -> str:
        """Builds and runs the ReAct agent."""
        try:
            agent = StandaloneReActAgent(
                llm=llm,
                tools=tools,
                verbose=True, # Consider making this configurable or linking to Langflow logging
                max_iterations=max_iterations,
                handle_parsing_errors=handle_parsing_errors,
                system_prompt=system_prompt, # Pass override if provided
                # callbacks=callbacks,
            )
            # AgentExecutor is built within the agent's run method now
            result = agent.run(input_value)
            self.status = f"Agent finished successfully. Result: {result[:50]}..."
            return result
        except Exception as e:
            self.status = f"Error running agent: {str(e)}"
            # Log the full error for debugging
            print(f"Error details: {e}")
            # Potentially re-raise or return an error message
            return f"Error: {str(e)}"
