"""Mock classes for testing."""

from unittest.mock import MagicMock
from typing import Any, Dict, List, Optional, Union

# Import everything from langchain-core for better compatibility
from langchain_core.tools import BaseTool
from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage


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
    """Mock LLM that implements all required abstract methods."""
    
    # Avoid using __init__ with BaseLanguageModel due to Pydantic validation
    response: str = "This is a mock response"
    predict_called: bool = False
    
    # We'll use a factory method instead of __init__
    @classmethod
    def create(cls, response: str = "This is a mock response"):
        """Create a MockLLM with the specified response."""
        instance = cls()
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(instance, "response", response)
        object.__setattr__(instance, "predict_called", False)
        return instance
        
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "mock"
        
    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        """Mock generation."""
        return {"generations": [[{"text": self.response}]]}
    
    async def _agenerate(self, prompts, stop=None, run_manager=None, **kwargs):
        """Mock async generation."""
        return self._generate(prompts, stop, run_manager, **kwargs)
        
    def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        """Mock generate from prompt."""
        return self._generate(prompts, stop, callbacks, **kwargs)
        
    async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        """Mock async generate from prompt."""
        return await self._agenerate(prompts, stop, callbacks, **kwargs)
    
    def predict(self, text: str, *, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Mock predict method."""
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, "predict_called", True)
        return self.response
        
    async def apredict(self, text: str, *, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Mock async predict method."""
        return self.predict(text, stop=stop, **kwargs)
    
    def predict_messages(
        self, messages: List[BaseMessage], *, stop: Optional[List[str]] = None, **kwargs
    ) -> AIMessage:
        """Mock predict messages method."""
        return AIMessage(content=self.response)
    
    async def apredict_messages(
        self, messages: List[BaseMessage], *, stop: Optional[List[str]] = None, **kwargs
    ) -> AIMessage:
        """Mock async predict messages method."""
        return self.predict_messages(messages, stop=stop, **kwargs)
        
    def invoke(
        self, input: Union[str, List[BaseMessage]], config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Union[str, BaseMessage]:
        """Mock invoke method."""
        if isinstance(input, str):
            return self.predict(input)
        else:
            return self.predict_messages(input)
            
    def bind_callbacks(self, callbacks: Optional[BaseCallbackManager]) -> BaseLanguageModel:
        """Bind callbacks to this LLM."""
        return self
