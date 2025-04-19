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
    repo_id="microsoft/phi-3-mini-4k-instruct",  # Or any other HF model
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
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

### Option 1: Docker Setup (Recommended)

This project uses Docker to handle dependency conflicts between core requirements and Langflow. The containerized setup provides consistent environments for development and testing.

#### Prerequisites
- Docker and Docker Compose installed on your system

#### Quick Start

1. Build the Docker images:
   ```bash
   make build
   ```

2. Run unit tests (no Langflow dependency):
   ```bash
   make unit-tests
   ```

3. Run integration tests (with Langflow):
   ```bash
   make integration-tests
   ```

4. Start the development environment with Langflow:
   ```bash
   make dev
   ```

5. Access the Langflow UI at http://localhost:7860

#### How Our Docker-Langflow Integration Works

We use a specialized Docker setup to properly install and run Langflow without dependency conflicts:

1. **Isolated Virtual Environments**: 
   - `/opt/venv-core`: Contains core dependencies for standalone functionality
   - `/opt/venv-langflow`: Contains Langflow and minimal dependencies

2. **Dependency Isolation**: Each environment has its own Python interpreter and packages, preventing conflicts

3. **Activation Scripts**: Simple scripts (`/activate-core.sh` and `/activate-langflow.sh`) handle environment switching

4. **Volume Mapping**: The `langflow_cache` volume persists Langflow settings between container restarts

5. **Environment Variables**: The container automatically sets `LANGFLOW_COMPONENT_PATH` to load our custom components

#### Environment Variables

Create a `.env` file with the following variables:
```
HUGGINGFACEHUB_API_TOKEN="your_huggingface_token"
```

### Option 2: Manual Setup

1. Activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the core dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. For Langflow integration, create a separate environment:
   ```bash
   python -m venv langflow-env
   source langflow-env/bin/activate  # On Windows: langflow-env\Scripts\activate
   pip install langflow
   pip install -e .
   ```

4. Set the environment variable to load our custom components:
   ```bash
   export LANGFLOW_COMPONENT_PATH=/path/to/project/src/integrations/langflow
   ```

5. Run Langflow:
   ```bash
   langflow run
   ```

6. Access the Langflow UI at http://localhost:7860

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

## Testing

The project includes comprehensive tests, which you can run using Docker with the included Makefile commands:

```bash
# Build the Docker images
make build

# Run all tests
make all-tests

# Unit tests only
make unit-tests

# Integration tests only
make integration-tests

# Run Hugging Face model integration tests specifically
make hf-model-tests
```

#### Manual Testing (Not Recommended)

While it's recommended to use the Docker-based testing setup to avoid dependency issues, you can run tests manually:

```bash
# Unit tests
python -m pytest tests/unit

# Integration tests (requires proper environment setup)
python -m pytest tests/integration
```

**Note**: For the integration tests to work, you need to have the `HUGGINGFACEHUB_API_TOKEN` environment variable set.

## Additional Resources

- Documentation: See `docs/` directory
- Sample Flows: Available in `reference_flows/`
- Docker Configuration: See `Dockerfile` and `docker-compose.yml`
