"""
Standalone Plan-Execute Agent implementation.

This module provides a Plan-and-Execute agent that works with
any LLM, including open-source models hosted on Hugging Face, without
requiring the Langflow framework.
"""

from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms.base import BaseLLM as BaseLanguageModel
from langchain.tools.base import BaseTool


class PlanStep:
    """A single step in a plan."""
    
    def __init__(self, description: str, status: str = "pending"):
        self.description = description
        self.status = status
        self.result = None


class StandalonePlanExecuteAgent:
    """Standalone implementation of the Plan-Execute Agent.
    
    This agent uses a two-stage approach: first planning steps, then executing them.
    Works with any LLM, including open-source models hosted on Hugging Face.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        system_prompt: str = None,
        planner_prompt: str = None,
        verbose: bool = True,
        max_iterations: int = 5,
        handle_parsing_errors: bool = True,
        callbacks: List[BaseCallbackHandler] = None,
    ):
        """Initialize the Plan-Execute Agent.
        
        Args:
            llm: The language model to use
            tools: List of tools available to the agent
            system_prompt: System instructions for the agent
            planner_prompt: Instructions for the planning phase
            verbose: Whether to print detailed logs
            max_iterations: Maximum number of plan steps to execute
            handle_parsing_errors: Whether to try recovering from parsing errors
            callbacks: Optional callbacks for tracking agent execution
        """
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that first plans what to do, then executes the plan step by step. "
            "You have access to tools that can help you complete tasks. "
            "Think carefully about the best way to accomplish the user's request."
        )
        self.planner_prompt = planner_prompt or (
            "Let's break down this task into a specific plan with clear steps. "
            "What is the best approach to accomplish this goal? "
            "Create a plan with no more than 5 steps."
        )
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.handle_parsing_errors = handle_parsing_errors
        self.callbacks = callbacks or []
    
    def create_plan(self, query: str) -> List[PlanStep]:
        """Create a plan for solving the given query.
        
        Args:
            query: The user's request
            
        Returns:
            A list of plan steps
        """
        # Validate tool names to avoid conflicts
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Tool names must be unique")
        
        # Create a planning prompt
        planner_template = PromptTemplate.from_template(
            f"{self.system_prompt}\n\n"
            f"{self.planner_prompt}\n\n"
            "For the following request, create a step-by-step plan:\n\n"
            "Request: {query}\n\n"
            "Respond with a numbered list of steps (no more than {max_steps}). "
            "Each step should be a clear, actionable instruction."
        )
        
        planner_chain = LLMChain(llm=self.llm, prompt=planner_template)
        
        # Generate the plan
        plan_text = planner_chain.run(query=query, max_steps=self.max_iterations)
        
        # Parse the plan into steps (simple parsing)
        steps = []
        for line in plan_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
                
            # Try to extract numbered steps
            if any(line.startswith(f"{i}.") for i in range(1, 10)):
                # Remove the number prefix
                step_text = line.split(".", 1)[1].strip()
                steps.append(PlanStep(description=step_text))
        
        # If parsing failed, create a single step with the entire plan
        if not steps:
            steps = [PlanStep(description=plan_text)]
            
        if self.verbose:
            print(f"📝 Created plan with {len(steps)} steps:")
            for i, step in enumerate(steps):
                print(f"  {i+1}. {step.description}")
                
        return steps
    
    def execute_step(self, step: PlanStep, tools: List[BaseTool]) -> str:
        """Execute a single step of the plan.
        
        Args:
            step: The plan step to execute
            tools: List of tools available for execution
            
        Returns:
            The result of executing the step
        """
        # Create a simple ReAct-like agent to execute this specific step
        from .react_agent import StandaloneReActAgent
        
        # Create a specialized system prompt for this step
        step_system_prompt = (
            f"{self.system_prompt}\n\n"
            f"You are currently executing this specific step of a larger plan:\n"
            f"'{step.description}'\n\n"
            "Focus only on completing this step using the available tools."
        )
        
        # Create a focused agent for this step
        step_agent = StandaloneReActAgent(
            llm=self.llm,
            tools=tools,
            system_prompt=step_system_prompt,
            verbose=self.verbose,
            max_iterations=3,  # Limit iterations for each step
            handle_parsing_errors=self.handle_parsing_errors,
            callbacks=self.callbacks
        )
        
        # Execute the step
        if self.verbose:
            print(f"🔄 Executing step: {step.description}")
        
        try:
            result = step_agent.run(step.description)
            step.status = "completed"
            step.result = result
            
            if self.verbose:
                print(f"✅ Step completed: {result[:100]}..." if len(result) > 100 else result)
                
            return result
        except Exception as e:
            step.status = "failed"
            step.result = f"Error: {str(e)}"
            
            if self.verbose:
                print(f"❌ Step failed: {str(e)}")
                
            return step.result
    
    def run(self, query: str) -> str:
        """Run the Plan-Execute agent on the given query.
        
        Args:
            query: The user's request
            
        Returns:
            The agent's final response
        """
        # Create the plan
        plan = self.create_plan(query)
        
        # Execute each step of the plan
        results = []
        for i, step in enumerate(plan):
            if self.verbose:
                print(f"\n📌 Step {i+1}/{len(plan)}: {step.description}")
                
            result = self.execute_step(step, self.tools)
            results.append(result)
        
        # Summarize the results
        summary_template = PromptTemplate.from_template(
            f"{self.system_prompt}\n\n"
            "Below is the original request, followed by the plan that was created and the results of executing each step:\n\n"
            "Request: {query}\n\n"
            "Plan and Results:\n{plan_results}\n\n"
            "Please provide a comprehensive summary that addresses the original request based on all the information gathered."
        )
        
        summary_chain = LLMChain(llm=self.llm, prompt=summary_template)
        
        # Format the plan and results
        plan_results = ""
        for i, (step, result) in enumerate(zip(plan, results)):
            plan_results += f"Step {i+1}: {step.description}\n"
            plan_results += f"Result: {result}\n\n"
        
        # Generate the final summary
        final_response = summary_chain.run(query=query, plan_results=plan_results)
        
        return final_response
    
    def __call__(self, query: str) -> str:
        """Allow the agent to be called directly."""
        return self.run(query)
