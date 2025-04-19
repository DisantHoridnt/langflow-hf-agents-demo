"""Tests for the Langflow ReAct Agent Component integration."""

import pytest
from unittest.mock import patch, MagicMock

from langchain.tools.base import BaseTool
from src.integrations.langflow import ReActAgentComponent


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


class TestReActAgentComponent:
    """Tests for the Langflow ReAct Agent Component."""

    def test_react_component_initialization(self):
        """Test component initialization with default parameters."""
        # Initialize the component
        component = ReActAgentComponent()
        
        # Check default values
        assert component.display_name == "ReAct Agent"
        assert "🤔" in component.icon
        assert component.group == "Agents"
        assert component.beta is False
        
        # Check required inputs exist
        input_names = [inp.name for inp in component.inputs if hasattr(inp, 'name')]
        assert "system_prompt" in input_names
        assert "verbose" in input_names
        assert "max_iterations" in input_names
        assert "handle_parsing_errors" in input_names
        
        # Check for memory input (for future v2 implementation)
        assert "memory" in input_names
    
    def test_set_attributes(self):
        """Test setting attributes on the component."""
        component = ReActAgentComponent()
        
        # Create mock objects
        mock_llm = MagicMock()
        mock_tools = [MockTool()]
        
        # Set attributes
        component.set(
            llm=mock_llm,
            tools=mock_tools,
            verbose=False,
            max_iterations=5,
            system_prompt="Custom system prompt"
        )
        
        # Verify attributes were set correctly
        assert component.llm == mock_llm
        assert component.tools == mock_tools
        assert component.verbose is False
        assert component.max_iterations == 5
        assert component.system_prompt == "Custom system prompt"
    
    def test_validate_tool_names(self):
        """Test tool name validation in the component."""
        component = ReActAgentComponent()
        
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
    
    def test_create_agent_runnable(self):
        """Test creation of the agent runnable."""
        component = ReActAgentComponent()
        
        # Mock the LLM and tools
        mock_llm = MagicMock()
        component.llm = mock_llm
        component.tools = [MockTool()]
        
        # Patch the create_react_agent function
        with patch("src.integrations.langflow.react_component.create_react_agent") as mock_create:
            mock_create.return_value = "mock_agent_runnable"
            
            # Call the method
            result = component.create_agent_runnable()
            
            # Check the result
            assert result == "mock_agent_runnable"
            mock_create.assert_called_once()
    
    def test_build_agent(self):
        """Test building the agent executor."""
        component = ReActAgentComponent()
        
        # Mock the required attributes and methods
        component.llm = MagicMock()
        component.tools = [MockTool()]
        component.handle_parsing_errors = True
        component.max_iterations = 7
        component.verbose = True
        
        # Patch methods
        with patch.object(component, "validate_tool_names") as mock_validate, \
             patch.object(component, "create_agent_runnable") as mock_create, \
             patch("src.integrations.langflow.react_component.AgentExecutor") as mock_executor:
            
            mock_create.return_value = "mock_agent"
            mock_executor.from_agent_and_tools.return_value = "mock_executor"
            
            # Call the method
            result = component.build_agent()
            
            # Check the results
            assert result == "mock_executor"
            mock_validate.assert_called_once()
            mock_create.assert_called_once()
            mock_executor.from_agent_and_tools.assert_called_once_with(
                agent="mock_agent",
                tools=[component.tools[0]],
                handle_parsing_errors=True,
                max_iterations=7,
                verbose=True,
                callbacks=[]
            )
    
    def test_build_with_memory(self):
        """Test building the agent with memory (for future v2)."""
        component = ReActAgentComponent()
        
        # Mock the required attributes and methods
        component.llm = MagicMock()
        component.tools = [MockTool()]
        component.memory = MagicMock()  # Add memory
        
        # Patch methods
        with patch.object(component, "validate_tool_names"), \
             patch.object(component, "create_agent_runnable") as mock_create, \
             patch("src.integrations.langflow.react_component.AgentExecutor") as mock_executor:
            
            mock_create.return_value = "mock_agent"
            mock_executor.from_agent_and_tools.return_value = "mock_executor"
            
            # Call the method
            component.build_agent()
            
            # Check that memory was included in the executor creation
            call_args = mock_executor.from_agent_and_tools.call_args[1]
            assert "memory" in call_args
            assert call_args["memory"] == component.memory
