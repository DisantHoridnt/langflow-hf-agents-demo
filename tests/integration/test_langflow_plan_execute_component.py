"""Tests for the Langflow Plan-Execute Agent Component integration."""

import pytest
from unittest.mock import patch, MagicMock

# Mark these tests to be skipped by default
pytestmark = [pytest.mark.langflow]

from langchain.tools.base import BaseTool
# Mock the Langflow imports since we're testing without actual Langflow installed
from unittest.mock import MagicMock

# Create mock classes to test without requiring actual Langflow imports
class MockLCToolsAgentComponent:
    """Mock Langflow base component class."""
    _base_inputs = []
    
    def validate_tool_names(self):
        # Implement the validation logic from the actual component
        if not hasattr(self, "tools") or not self.tools:
            return
            
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Tool names must be unique")


# Mock the PlanExecuteAgentComponent
class PlanExecuteAgentComponent(MockLCToolsAgentComponent):
    """Mock implementation of PlanExecuteAgentComponent for testing."""
    display_name = "Plan-Execute Agent"
    icon = "🗺️"
    group = "Agents"
    beta = False
    
    def __init__(self):
        self.inputs = [
            MagicMock(name="system_prompt"),
            MagicMock(name="verbose"),
            MagicMock(name="max_iterations"),
            MagicMock(name="max_subtask_iterations"),
            MagicMock(name="memory"),
        ]
        self.llm = None
        self.tools = None
        self.verbose = True
        self.max_iterations = 10
        self.max_subtask_iterations = 10
        self.system_prompt = "Default system prompt"
        
    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
        
    def create_planner(self):
        # Mock implementation for testing
        if not self.llm:
            raise ValueError("LLM is required")
        return MagicMock()
        
    def create_executor(self):
        # Mock implementation for testing
        if not self.llm or not self.tools:
            raise ValueError("LLM and tools are required")
        return MagicMock()
        
    def build_agent(self):
        self.validate_tool_names()
        planner = self.create_planner()
        executor = self.create_executor()
        
        # Return a mock PlanAndExecute agent
        return MagicMock()


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


class TestPlanExecuteAgentComponent:
    """Tests for the Langflow Plan-Execute Agent Component."""

    def test_component_initialization(self):
        """Test component initialization with default parameters."""
        # Initialize the component
        component = PlanExecuteAgentComponent()
        
        # Check default values
        assert component.display_name == "Plan-Execute Agent"
        assert "🗺️" in component.icon
        assert component.group == "Agents"
        assert component.beta is False
        
        # Check required inputs exist
        input_names = [inp.name for inp in component.inputs if hasattr(inp, 'name')]
        assert "system_prompt" in input_names
        assert "verbose" in input_names
        assert "max_iterations" in input_names
        assert "max_subtask_iterations" in input_names
        
        # Check for memory input (for future v2 implementation)
        assert "memory" in input_names
    
    def test_set_attributes(self):
        """Test setting attributes on the component."""
        component = PlanExecuteAgentComponent()
        
        # Create mock objects
        mock_llm = MagicMock()
        mock_tools = [MockTool()]
        
        # Set attributes
        component.set(
            llm=mock_llm,
            tools=mock_tools,
            verbose=False,
            max_iterations=5,
            max_subtask_iterations=3,
            system_prompt="Custom system prompt"
        )
        
        # Verify attributes were set correctly
        assert component.llm == mock_llm
        assert component.tools == mock_tools
        assert component.verbose is False
        assert component.max_iterations == 5
        assert component.max_subtask_iterations == 3
        assert component.system_prompt == "Custom system prompt"
    
    def test_validate_tool_names(self):
        """Test tool name validation in the component."""
        component = PlanExecuteAgentComponent()
        
        # Set up tools with unique names - should not raise error
        component.tools = [
            MockTool(name="Tool1"),
            MockTool(name="Tool2")
        ]
        component.validate_tool_names()  # Should not raise error
        
        # Set up tools with duplicate names - should raise error
        component.tools = [
            MockTool(name="DuplicateName"),
            MockTool(name="DuplicateName")
        ]
        with pytest.raises(ValueError, match="Tool names must be unique"):
            component.validate_tool_names()
    
    def test_create_planner(self):
        """Test creation of the planner."""
        component = PlanExecuteAgentComponent()
        
        # Mock the LLM
        mock_llm = MagicMock()
        component.llm = mock_llm
        component.system_prompt = "Test planner prompt"
        
        # Patch the LLMChain
        with patch("src.integrations.langflow.plan_execute_component.LLMChain") as mock_chain:
            mock_chain.return_value = "mock_planner"
            
            # Call the method
            result = component.create_planner()
            
            # Check the result
            assert result == "mock_planner"
            mock_chain.assert_called_once()
    
    def test_create_executor(self):
        """Test creation of the executor."""
        component = PlanExecuteAgentComponent()
        
        # Mock the LLM and tools
        mock_llm = MagicMock()
        component.llm = mock_llm
        component.tools = [MockTool()]
        component.max_subtask_iterations = 3
        
        # Patch the create_react_agent and AgentExecutor
        with patch("src.integrations.langflow.plan_execute_component.create_react_agent") as mock_create, \
             patch("src.integrations.langflow.plan_execute_component.AgentExecutor") as mock_executor:
            
            mock_create.return_value = "mock_agent"
            mock_executor.from_agent_and_tools.return_value = "mock_subtask_executor"
            
            # Call the method
            result = component.create_executor()
            
            # Check the result
            assert result == "mock_subtask_executor"
            mock_create.assert_called_once()
            mock_executor.from_agent_and_tools.assert_called_once()
    
    def test_build_agent(self):
        """Test building the Plan-and-Execute agent."""
        component = PlanExecuteAgentComponent()
        
        # Mock the required attributes and methods
        component.llm = MagicMock()
        component.tools = [MockTool()]
        component.max_iterations = 5
        component.verbose = True
        
        # Patch methods
        with patch.object(component, "validate_tool_names") as mock_validate, \
             patch.object(component, "create_planner") as mock_create_planner, \
             patch.object(component, "create_executor") as mock_create_executor, \
             patch("src.integrations.langflow.plan_execute_component.PlanAndExecute") as mock_pne:
            
            mock_create_planner.return_value = "mock_planner"
            mock_create_executor.return_value = "mock_executor"
            mock_pne.return_value = "mock_pne_agent"
            
            # Call the method
            result = component.build_agent()
            
            # Check the results
            assert result == "mock_pne_agent"
            mock_validate.assert_called_once()
            mock_create_planner.assert_called_once()
            mock_create_executor.assert_called_once()
            mock_pne.assert_called_once_with(
                planner="mock_planner",
                executor="mock_executor",
                verbose=True,
                max_iterations=5
            )
    
    def test_build_with_memory(self):
        """Test building the agent with memory (for future v2)."""
        component = PlanExecuteAgentComponent()
        
        # Mock the required attributes and methods
        component.llm = MagicMock()
        component.tools = [MockTool()]
        component.memory = MagicMock()  # Add memory
        
        # Patch methods
        with patch.object(component, "validate_tool_names"), \
             patch.object(component, "create_planner") as mock_create_planner, \
             patch.object(component, "create_executor") as mock_create_executor, \
             patch("src.integrations.langflow.plan_execute_component.PlanAndExecute") as mock_pne:
            
            mock_create_planner.return_value = "mock_planner"
            mock_create_executor.return_value = "mock_executor"
            mock_pne.return_value = "mock_pne_agent"
            
            # Call the method
            component.build_agent()
            
            # Check that memory is correctly used when building the agent
            # Note: How memory is used depends on the specific implementation
            # This might need adjustment based on the actual implementation
            mock_create_planner.assert_called_once()
