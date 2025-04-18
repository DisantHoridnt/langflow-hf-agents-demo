"""
Tests for the ReAct Agent Component.
"""

import pytest
from unittest.mock import patch, MagicMock

from langflow.custom import CustomComponent

from ..react_agent import ReActAgentComponent
from .test_utils import create_mock_components


def test_react_agent_initialization():
    """Test that the ReAct agent component initializes correctly."""
    # Initialize the component
    agent_component = ReActAgentComponent()
    
    # Check that it was initialized correctly
    assert agent_component.display_name == "ReAct Agent"
    assert agent_component.name == "ReActAgentComponent"
    assert agent_component.group == "Agents"
    assert "system_prompt" in [input_.name for input_ in agent_component.inputs]
    assert "verbose" in [input_.name for input_ in agent_component.inputs]
    assert "max_iterations" in [input_.name for input_ in agent_component.inputs]


def test_react_agent_set_attributes():
    """Test that the set method correctly sets attributes."""
    # Initialize the component
    agent_component = ReActAgentComponent()
    
    # Use set method
    llm, tools = create_mock_components()
    agent_component.set(
        llm=llm,
        tools=tools,
        system_prompt="Test prompt",
        verbose=True,
        max_iterations=5
    )
    
    # Check that attributes were set correctly
    assert agent_component.llm == llm
    assert agent_component.tools == tools
    assert agent_component.system_prompt == "Test prompt"
    assert agent_component.verbose is True
    assert agent_component.max_iterations == 5


def test_react_agent_create_agent_runnable():
    """Test that create_agent_runnable returns a valid agent."""
    # Initialize the component
    agent_component = ReActAgentComponent()
    
    # Set required attributes
    llm, tools = create_mock_components()
    agent_component.set(
        llm=llm,
        tools=tools,
        system_prompt="Test prompt",
        verbose=True,
        max_iterations=5
    )
    
    # Create the agent runnable
    agent = agent_component.create_agent_runnable()
    
    # Check that it returned a valid object
    assert agent is not None


def test_react_agent_validation_errors():
    """Test that the component raises appropriate errors when validations fail."""
    # Initialize the component
    agent_component = ReActAgentComponent()
    
    # Test missing LLM
    with pytest.raises(ValueError, match="Expected llm to be a BaseLanguageModel"):
        agent_component.set(tools=[])
        agent_component.create_agent_runnable()
    
    # Test missing tools
    llm, _ = create_mock_components()
    with pytest.raises(ValueError, match="Tools are required"):
        agent_component.set(llm=llm, tools=None)
        agent_component.create_agent_runnable()


@patch("langchain.agents.AgentExecutor.from_agent_and_tools")
def test_react_agent_build(mock_from_agent_and_tools):
    """Test the build_agent method that creates the AgentExecutor."""
    # Setup mock
    mock_executor = MagicMock()
    mock_from_agent_and_tools.return_value = mock_executor
    
    # Initialize and set up the component
    agent_component = ReActAgentComponent()
    llm, tools = create_mock_components()
    agent_component.set(
        llm=llm,
        tools=tools,
        system_prompt="Test prompt",
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    # Mock validate_tool_names to avoid dependency issues
    agent_component.validate_tool_names = MagicMock()
    
    # Call build_agent
    executor = agent_component.build_agent()
    
    # Verify the results
    assert executor == mock_executor
    agent_component.validate_tool_names.assert_called_once()
    mock_from_agent_and_tools.assert_called_once()
    
    # Check that the correct parameters were passed
    _, kwargs = mock_from_agent_and_tools.call_args
    assert kwargs["tools"] == tools
    assert kwargs["handle_parsing_errors"] is True
    assert kwargs["max_iterations"] == 5
    assert kwargs["verbose"] is True
