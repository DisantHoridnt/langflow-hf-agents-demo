# HF‑Native Agents

A professional implementation of ReAct & Plan‑and‑Execute Agent Components for both Langflow integration and standalone usage with Hugging Face models.

## Overview

This package provides two distinct implementations:

1. **Standalone Agents** - Use with any Python project:
   - `StandaloneReActAgent` - Implements the ReAct (Reasoning and Acting) pattern
   - `StandalonePlanExecuteAgent` - Implements the Plan-and-Execute pattern

2. **Langflow Components** - For low-code flow creation in Langflow:
   - `ReActAgentComponent` - Appears in the Langflow UI as "ReAct Agent"
   - `PlanExecuteAgentComponent` - Appears in the Langflow UI as "Plan-Execute Agent"

Both implementations work with any LLM, including open-source models hosted on Hugging Face, without requiring proprietary APIs.

## Installation

### Usage

### Standalone Usage

```python
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from hf_agents import StandaloneReActAgent

# Load environment variables
load_dotenv()

# Create an LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Create some tools
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wikipedia_tool.name = "Lookup"  # ReAct agent expects specific tool names

# Create the agent
agent = StandaloneReActAgent(llm=llm, tools=[wikipedia_tool])

# Run the agent
result = agent.run("What is the capital of France?")
print(result)
```

### Langflow Integration

To use the Langflow components:

1. Install Langflow in a separate environment:
   ```bash
   pip install langflow
   ```

2. Set the component path:
   ```bash
   export LANGFLOW_COMPONENT_PATH=/path/to/langflow-hf-agents-demo/src/langflow_components
   ```

3. Start Langflow:
   ```bash
   langflow run
   ```

4. Import the reference flows from `reference_flows/`

## Project Structure

```
├── src/                        # Source code
│   ├── core/                   # Core implementations
│   │   └── agents/             # Agent implementations
│   │       ├── react.py         # ReAct agent
│   │       └── plan_execute.py  # Plan-Execute agent
│   └── integrations/           # Framework integrations
│       └── langflow/           # Langflow components
│           ├── react_component.py    # ReAct component for Langflow
│           └── plan_execute_component.py # Plan-Execute component
├── reference_flows/            # Reference Langflow configurations
├── tests/                      # Test suite
├── docs/                       # Documentation
└── setup.py                    # Package installation
```

## Setup Instructions

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set the environment variable to load our custom components:
   ```bash
   export LANGFLOW_COMPONENT_PATH=$PWD/custom_components
   ```

4. Run Langflow:
   ```bash
   langflow run
   ```

5. Access the Langflow UI at http://localhost:7860

## Demo Scenarios

### Scenario 1: ReAct Agent with Calculator

1. Import the sample flow `custom_components/hf_native_agents/sample_flows/react_agent_flow.json`
2. Add your Hugging Face API token to the HuggingFace model component
3. Ask questions that require calculation, like:
   - "What is the square root of 144 divided by 3?"
   - "If I have 5 apples and give away 2, then buy 3 more, how many do I have?"

### Scenario 2: Plan-and-Execute with Research

1. Import the sample flow `custom_components/hf_native_agents/sample_flows/plan_execute_agent_flow.json`
2. Add your Hugging Face API token and SerpAPI key to the respective components
3. Ask multi-step questions that require both search and calculation, like:
   - "What is the population of France, and what percentage is it of the world population?"
   - "Find the GDP of Germany and convert it to Japanese Yen."

## Key Selling Points

1. **Open-source LLM Compatibility**: Works with Mistral, Llama, Phi and other models
2. **Cost-effective**: No dependency on expensive API calls
3. **Privacy and control**: Can work with self-hosted models
4. **Full tool support**: Uses standard LangChain tools with no code changes
5. **Flexible architecture**: Supports both ReAct and Plan-and-Execute patterns

## Additional Resources

- Documentation: See `custom_components/hf_native_agents/README.md` 
- Sample Flows: Available in `custom_components/hf_native_agents/sample_flows/`
- Test Suite: Run with `pytest custom_components/hf_native_agents/tests/`
