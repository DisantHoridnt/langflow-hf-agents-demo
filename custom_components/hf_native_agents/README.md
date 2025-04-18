# HF-Native ReAct & Plan-and-Execute Agent Components for Langflow

This package provides two custom Langflow components that enable using any LLM (including open-source models served from Hugging Face Inference Endpoints) with LangChain's ReAct and Plan-and-Execute agent architectures.

## Components

### 1. ReActAgentComponent (🤔)

The ReAct (Reasoning and Acting) agent uses a reasoning process to determine what actions to take based on the user's input. It works with any LLM, including open-source models, and doesn't require the tool-calling capabilities that OpenAI-style agents need.

### 2. PlanExecuteAgentComponent (🗺️)

The Plan-and-Execute agent uses a two-stage process:
1. First, it creates a plan with specific steps to accomplish the task
2. Then, it executes each step individually, using tools when necessary

This approach is particularly effective for complex tasks that benefit from breaking down the problem into smaller steps.

## Installation

### 1. Set up the components path

Add the parent directory of this package to your `LANGFLOW_COMPONENT_PATH` environment variable:

```bash
export LANGFLOW_COMPONENT_PATH=/path/to/custom_components
```

Or add it to your `.env` file if you're using one:

```
LANGFLOW_COMPONENT_PATH=/path/to/custom_components
```

### 2. Install dependencies

Make sure you have the required dependencies:

```bash
pip install langchain langchain-experimental
```

## Usage

After installation, the components will appear in the Langflow UI under the "Agents" tab. To create a flow with these components:

1. Add a "ReAct Agent" (🤔) or "Plan-Execute Agent" (🗺️) to your canvas
2. Connect a Hugging Face model or any other LLM to the agent's `llm` input port
3. Connect one or more tools to the agent's `tools` input port
4. Connect a ChatInput component to the agent's `input_value` input
5. Connect the agent to a ChatOutput component

### Sample Flows

Two example flows are provided to help you get started:

1. **React Agent Flow**: `sample_flows/react_agent_flow.json` - Demonstrates the ReAct agent with Calculator and Search tools using Mistral-7B
2. **Plan-Execute Agent Flow**: `sample_flows/plan_execute_agent_flow.json` - Demonstrates the Plan-Execute agent with SerpAPI and Calculator tools using Phi-3-mini

### Example Flow Structure

```
+-------------------+     tools[]    +---------------+
| ChatInput (msg)   | ─────────────▶ |   ReAct Agent |
+-------------------+                |               |
                                     |   ┌────────┐  |
                      HuggingFace ──▶|   │   LLM  │  |
                                     |   └────────┘  |
                                     +-------│-------+
                                             ▼
                                     +---------------+
                                     | ChatOutput    |
                                     +---------------+
```

## Model Compatibility

These components have been tested with:
- Mistral-7B-Instruct
- Phi-3-mini
- Llama-3.1-8B-Instruct

## Future Enhancements (v2)

- **Memory Integration**: The components are already prepared with optional memory inputs for future implementation
- **Streaming Optimization**: Improving token streaming performance
- **GUI Visualization**: Enhanced visual representations of planning and execution stages

## Features

- **Icon Customization**: Both components use distinctive emojis (🤔 for ReAct, 🗺️ for Plan-Execute) for easy identification
- **Memory Support Preparation**: Both components have optional memory inputs ready for v2 implementation
- **Sample Flows**: Ready-to-use example flows for both agent types with recommended tool configurations

## Limitations

- First token generation may take up to 15 seconds depending on model size and network conditions
- Complex reasoning chains may require higher token limits for larger models
- The agent will only use tools that are explicitly provided to it

## Troubleshooting

- If the agent isn't showing up in the UI, verify your `LANGFLOW_COMPONENT_PATH` is set correctly
- For parsing errors, try increasing the max_iterations parameter
- Models with limited context windows may struggle with complex multi-step reasoning tasks
