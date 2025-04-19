"""Test configuration and fixtures for the HF Agents tests."""

import sys
import pytest
from unittest.mock import MagicMock

# Create mocks for external dependencies that might not be available
# This allows testing core functionality without requiring all dependencies

class MockLangflowImports:
    """Mock Langflow imports for testing without requiring actual Langflow."""
    
    class BaseComponent:
        """Mock base component."""
        _base_inputs = []
    
    class LCToolsAgentComponent(BaseComponent):
        """Mock Langflow agent component."""
        
        def get_langchain_callbacks(self):
            """Return empty callbacks list."""
            return []
        
        def validate_tool_names(self):
            """Validate tool names are unique."""
            if not hasattr(self, "tools") or not self.tools:
                return
                
            tool_names = [tool.name for tool in self.tools]
            if len(tool_names) != len(set(tool_names)):
                raise ValueError("Tool names must be unique")


# Patch imports before tests run
@pytest.fixture(scope="session", autouse=True)
def patch_imports():
    """Patch imports for testing without requiring all dependencies."""
    # Create mock modules
    mock_modules = {
        "langflow": MagicMock(),
        "langflow.base": MagicMock(),
        "langflow.base.agents": MagicMock(),
        "langflow.base.agents.agent": MagicMock(),
        "langflow.field_typing": MagicMock(),
        "langflow.io": MagicMock(),
    }
    
    # Set up LCToolsAgentComponent mock
    mock_modules["langflow.base.agents.agent"].LCToolsAgentComponent = MockLangflowImports.LCToolsAgentComponent
    
    # Set up other mocks
    mock_modules["langflow.field_typing"].LanguageModel = MagicMock()
    mock_modules["langflow.field_typing"].Tool = MagicMock()
    mock_modules["langflow.field_typing"].BaseMemory = MagicMock()
    
    mock_modules["langflow.io"].BoolInput = MagicMock()
    mock_modules["langflow.io"].IntInput = MagicMock()
    mock_modules["langflow.io"].MultilineInput = MagicMock()
    
    # Add them to sys.modules
    for name, mock in mock_modules.items():
        if name not in sys.modules:
            sys.modules[name] = mock
    
    yield
    
    # Clean up modules after tests
    for name in mock_modules:
        if name in sys.modules:
            del sys.modules[name]
