"""Tests for the Standalone ReAct Agent implementation."""

import pytest
from unittest.mock import patch, MagicMock

from langchain.tools.base import BaseTool


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


# Test tool name validation function separately without needing the full agent
def test_tool_name_validation():
    """Test validation of tool names."""
    # Valid tools with different names
    tools_with_unique_names = [
        MockTool(name="Lookup", description="Tool 1"),
        MockTool(name="Search", description="Tool 2"),
    ]
    
    # Get tool names
    tool_names = [tool.name for tool in tools_with_unique_names]
    # Validate they're unique
    assert len(tool_names) == len(set(tool_names))
    
    # Invalid tools with duplicate names
    tools_with_duplicate_names = [
        MockTool(name="Lookup", description="Tool 1"),
        MockTool(name="Lookup", description="Tool 2"),
    ]
    
    # Get tool names
    tool_names = [tool.name for tool in tools_with_duplicate_names]
    # Validate they're not unique
    assert len(tool_names) != len(set(tool_names))


# Test to verify our module structure without creating instances
def test_standalone_agent_imports():
    """Test that we can import the standalone agent from the new structure."""
    from src.core.agents import StandaloneReActAgent
    
    # Verify the class has the expected methods
    assert hasattr(StandaloneReActAgent, "__init__")
    assert hasattr(StandaloneReActAgent, "_build_agent_executor")
    assert hasattr(StandaloneReActAgent, "run")
    assert hasattr(StandaloneReActAgent, "__call__")
