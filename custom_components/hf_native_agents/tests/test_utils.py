"""
Test utilities for HF Native Agent Components.

Provides mock objects and helper functions for testing.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool


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


class MockLLM(BaseLanguageModel):
    """Mock LLM for testing purposes."""
    
    def __init__(self, responses: Optional[List[str]] = None):
        """Initialize with optional predetermined responses."""
        self.responses = responses or ["This is a mock response"]
        self.response_index = 0
    
    def get_response(self) -> str:
        """Get next response or cycle back to the first."""
        if self.response_index >= len(self.responses):
            self.response_index = 0
        response = self.responses[self.response_index]
        self.response_index += 1
        return response
    
    def invoke(self, prompt: Any, **kwargs: Any) -> Any:
        """Mock invoke method that returns a predetermined response."""
        return {"content": self.get_response()}
    
    def generate(self, prompts: List[str], **kwargs: Any) -> Any:
        """Mock generate method that returns predetermined responses."""
        generations = []
        for _ in prompts:
            generations.append([{"text": self.get_response()}])
        return MagicMock(generations=generations)
    
    async def agenerate(self, prompts: List[str], **kwargs: Any) -> Any:
        """Mock async generate method."""
        return self.generate(prompts, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "mock"


def create_mock_components():
    """Create mock components for testing."""
    # Create mock LLM
    llm = MockLLM([
        "I'll solve this by: 1. First checking X, 2. Then doing Y",
        "I used the tool and found: the answer is 42",
        "The final answer is: 42"
    ])
    
    # Create mock tools
    tools = [
        MockTool(name="calculator", description="Useful for calculations"),
        MockTool(name="search", description="Search for information"),
    ]
    
    return llm, tools
