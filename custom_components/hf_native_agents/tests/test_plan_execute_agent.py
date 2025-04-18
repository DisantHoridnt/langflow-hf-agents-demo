"""
Tests for the Plan-Execute Agent Component.
"""

import pytest
from unittest.mock import patch, MagicMock

from langflow.custom import CustomComponent

from ..plan_execute_agent import PlanExecuteAgentComponent
from .test_utils import create_mock_components


def test_plan_execute_agent_initialization():
    """Test that the Plan-Execute agent component initializes correctly."""
    # Initialize the component
    agent_component = PlanExecuteAgentComponent()
    
    # Check that it was initialized correctly
    assert agent_component.display_name == "Plan-Execute Agent"
    assert agent_component.name == "PlanExecuteAgentComponent"
    assert agent_component.group == "Agents"
    assert "system_prompt" in [input_.name for input_ in agent_component.inputs]
    assert "planner_prompt" in [input_.name for input_ in agent_component.inputs]
    assert "verbose" in [input_.name for input_ in agent_component.inputs]
    assert "max_iterations" in [input_.name for input_ in agent_component.inputs]
    assert "max_execution_iterations" in [input_.name for input_ in agent_component.inputs]


def test_plan_execute_agent_set_attributes():
    """Test that the set method correctly sets attributes."""
    # Initialize the component
    agent_component = PlanExecuteAgentComponent()
    
    # Use set method
    llm, tools = create_mock_components()
    agent_component.set(
        llm=llm,
        tools=tools,
        system_prompt="Test system prompt",
        planner_prompt="Test planner prompt",
        verbose=True,
        max_iterations=5,
        max_execution_iterations=3
    )
    
    # Check that attributes were set correctly
    assert agent_component.llm == llm
    assert agent_component.tools == tools
    assert agent_component.system_prompt == "Test system prompt"
    assert agent_component.planner_prompt == "Test planner prompt"
    assert agent_component.verbose is True
    assert agent_component.max_iterations == 5
    assert agent_component.max_execution_iterations == 3


def test_plan_execute_agent_validation_errors():
    """Test that the component raises appropriate errors when validations fail."""
    # Initialize the component
    agent_component = PlanExecuteAgentComponent()
    
    # Test missing LLM
    with pytest.raises(ValueError, match="Expected llm to be a BaseLanguageModel"):
        agent_component.set(tools=[])
        agent_component.create_agent_runnable()
    
    # Test missing tools
    llm, _ = create_mock_components()
    with pytest.raises(ValueError, match="Tools are required"):
        agent_component.set(llm=llm, tools=None)
        agent_component.create_agent_runnable()


@patch("langchain_experimental.plan_and_execute.load_chat_planner")
@patch("langchain_experimental.plan_and_execute.load_agent_executor")
@patch("langchain_experimental.plan_and_execute.PlanAndExecute")
def test_plan_execute_agent_create_runnable(
    mock_plan_and_execute, mock_load_agent_executor, mock_load_chat_planner
):
    """Test that create_agent_runnable creates the PlanAndExecute agent correctly."""
    # Setup mocks
    mock_planner = MagicMock()
    mock_executor = MagicMock()
    mock_agent = MagicMock()
    
    mock_load_chat_planner.return_value = mock_planner
    mock_load_agent_executor.return_value = mock_executor
    mock_plan_and_execute.return_value = mock_agent
    
    # Initialize and set up the component
    agent_component = PlanExecuteAgentComponent()
    llm, tools = create_mock_components()
    agent_component.set(
        llm=llm,
        tools=tools,
        system_prompt="Test system prompt",
        planner_prompt="Test planner prompt",
        verbose=True,
        max_iterations=5,
        max_execution_iterations=3,
        handle_parsing_errors=True
    )
    
    # Create agent runnable
    agent = agent_component.create_agent_runnable()
    
    # Verify results
    assert agent == mock_agent
    
    # Verify planner creation
    mock_load_chat_planner.assert_called_once()
    planner_args, planner_kwargs = mock_load_chat_planner.call_args
    assert planner_kwargs["llm"] == llm
    assert "prompt" in planner_kwargs
    assert planner_kwargs["planner_prompt"] == "Test planner prompt"
    
    # Verify executor creation
    mock_load_agent_executor.assert_called_once()
    executor_args, executor_kwargs = mock_load_agent_executor.call_args
    assert executor_kwargs["llm"] == llm
    assert executor_kwargs["tools"] == tools
    assert "prompt" in executor_kwargs
    assert executor_kwargs["verbose"] is True
    assert executor_kwargs["max_iterations"] == 3
    assert executor_kwargs["handle_parsing_errors"] is True
    
    # Verify PlanAndExecute creation
    mock_plan_and_execute.assert_called_once()
    pe_args, pe_kwargs = mock_plan_and_execute.call_args
    assert pe_kwargs["planner"] == mock_planner
    assert pe_kwargs["executor"] == mock_executor
    assert pe_kwargs["verbose"] is True


@patch("langchain.agents.AgentExecutor.from_agent_and_tools")
def test_plan_execute_agent_build(mock_from_agent_and_tools):
    """Test the build_agent method that creates the AgentExecutor."""
    # Setup mock
    mock_executor = MagicMock()
    mock_from_agent_and_tools.return_value = mock_executor
    
    # Initialize and set up the component
    agent_component = PlanExecuteAgentComponent()
    
    # Mock create_agent_runnable to return a mock agent
    mock_agent = MagicMock()
    agent_component.create_agent_runnable = MagicMock(return_value=mock_agent)
    
    # Mock validate_tool_names to avoid dependency issues
    agent_component.validate_tool_names = MagicMock()
    
    # Set required attributes
    llm, tools = create_mock_components()
    agent_component.set(
        llm=llm,
        tools=tools,
        system_prompt="Test system prompt",
        planner_prompt="Test planner prompt",
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    # Call build_agent
    executor = agent_component.build_agent()
    
    # Verify the results
    assert executor == mock_executor
    agent_component.validate_tool_names.assert_called_once()
    agent_component.create_agent_runnable.assert_called_once()
    mock_from_agent_and_tools.assert_called_once()
    
    # Check that the correct parameters were passed
    args, kwargs = mock_from_agent_and_tools.call_args
    assert kwargs["agent"] == mock_agent
    assert kwargs["tools"] == tools
    assert kwargs["verbose"] is True
    assert kwargs["max_iterations"] == 5
    assert kwargs["handle_parsing_errors"] is True
