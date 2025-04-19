"""Tests for the Standalone Plan-Execute Agent implementation."""

import pytest
from unittest.mock import patch, MagicMock

from langchain.tools.base import BaseTool
from src.core.agents import StandalonePlanExecuteAgent
from src.core.agents.plan_execute import PlanStep


class MockTool(BaseTool):
    """Mock tool for testing purposes."""
    
    name: str = "mock_tool"
    description: str = "A mock tool for testing"
    
    def _run(self, query: str) -> str:
        """Run the tool."""
        return f"Mock result for: {query}"
    
    async def _arun(self, query: str) -> str:
        """Run the tool asynchronously."""
        return f"Mock async result for: {query}"


class TestPlanExecuteAgent:
    """Test cases for the Plan-Execute Agent."""
    
    def test_planstep_initialization(self):
        """Test the PlanStep class initializes correctly."""
        step = PlanStep("Test step")
        assert step.description == "Test step"
        assert step.status == "pending"
        assert step.result is None
        
        # Test with custom status
        step = PlanStep("Test step", status="completed")
        assert step.status == "completed"
    
    def test_plan_execute_agent_initialization(self):
        """Test agent initialization with default parameters."""
        mock_llm = MagicMock()
        tools = [MockTool()]
        
        agent = StandalonePlanExecuteAgent(llm=mock_llm, tools=tools)
        
        assert agent.llm == mock_llm
        assert agent.tools == tools
        assert agent.verbose is True  # Default value
        assert agent.max_iterations == 10  # Default value
        assert agent.max_subtask_iterations == 10  # Default value
    
    def test_plan_execute_agent_custom_params(self):
        """Test agent initialization with custom parameters."""
        mock_llm = MagicMock()
        tools = [MockTool()]
        
        agent = StandalonePlanExecuteAgent(
            llm=mock_llm,
            tools=tools,
            verbose=False,
            max_iterations=5,
            max_subtask_iterations=3,
            system_prompt="Custom prompt"
        )
        
        assert agent.llm == mock_llm
        assert agent.tools == tools
        assert agent.verbose is False
        assert agent.max_iterations == 5
        assert agent.max_subtask_iterations == 3
        assert "Custom prompt" in agent.system_prompt
    
    def test_create_plan(self):
        """Test plan creation."""
        mock_llm = MagicMock()
        # Mock the LLM's predict method to return a formatted plan
        mock_llm.predict.return_value = "Step 1: Research information\nStep 2: Analyze data\nStep 3: Draw conclusions"
        
        agent = StandalonePlanExecuteAgent(llm=mock_llm, tools=[MockTool()])
        
        # Patch the _create_plan_chain method to return our mock LLM chain
        with patch.object(StandalonePlanExecuteAgent, '_create_plan_chain') as mock_create_chain:
            mock_chain = MagicMock()
            mock_chain.run.return_value = mock_llm.predict.return_value
            mock_create_chain.return_value = mock_chain
            
            plan = agent.create_plan("What is the capital of France?")
            
            assert len(plan) == 3
            assert plan[0].description == "Research information"
            assert plan[1].description == "Analyze data"
            assert plan[2].description == "Draw conclusions"
    
    def test_execute_step_with_tools(self):
        """Test step execution with tools."""
        mock_llm = MagicMock()
        mock_tool = MockTool()
        
        agent = StandalonePlanExecuteAgent(llm=mock_llm, tools=[mock_tool])
        
        # Create a test plan step
        step = PlanStep("Look up information about France")
        
        # Patch the _create_execution_chain method
        with patch.object(StandalonePlanExecuteAgent, '_create_execution_chain') as mock_create_chain:
            mock_chain = MagicMock()
            mock_chain.run.return_value = "Paris is the capital of France"
            mock_create_chain.return_value = mock_chain
            
            result = agent.execute_step(step, [mock_tool])
            
            assert result == "Paris is the capital of France"
            assert step.status == "completed"
            assert step.result == "Paris is the capital of France"
    
    def test_run_validates_tools(self):
        """Test that the run method validates tool names."""
        mock_llm = MagicMock()
        # Create tools with duplicate names
        tools = [
            MockTool(name="Lookup"),
            MockTool(name="Lookup")  # Duplicate name
        ]
        
        agent = StandalonePlanExecuteAgent(llm=mock_llm, tools=tools)
        
        # Using the duplicate tools should raise a ValueError
        with pytest.raises(ValueError, match="Tool names must be unique"):
            agent.run("Tell me about Paris")
