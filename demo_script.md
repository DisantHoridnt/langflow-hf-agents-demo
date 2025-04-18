# HF-Native Agents Demo Script

## Introduction (2 minutes)

"Today I'll demonstrate the new HF-Native Agent components we've developed for Langflow. These components solve a critical problem: enabling the use of open-source LLMs like Mistral and Llama with the powerful agent capabilities that were previously only available with OpenAI models.

The components implement both ReAct and Plan-and-Execute patterns, working with any Hugging Face-hosted model and any standard LangChain tools."

## Demo Setup (3 minutes)

1. Show the terminal with the demo environment:
   ```bash
   # Activate environment and set component path
   source venv/bin/activate
   export LANGFLOW_COMPONENT_PATH=$PWD/custom_components
   
   # Start Langflow
   langflow run
   ```

2. Open the browser to Langflow UI at http://localhost:7860

3. "First I'll show you the ReAct Agent component that we developed. This uses reasoning and acting cycles to solve problems step by step with any LLM."

## Demo 1: ReAct Agent with Calculator (5 minutes)

1. Import the sample flow:
   - Click "Import"
   - Select `react_agent_flow.json`
   - Show the components in the canvas

2. Highlight component connections:
   - "Notice how the HuggingFace model connects directly to our ReAct Agent"
   - "We've connected multiple tools - a Calculator and Search Tool"
   - "The agent can dynamically use any tool we connect to it"

3. Add your Hugging Face API token:
   - Click on the HuggingFace model
   - Enter your API token
   - "We're using Mistral-7B here, but this works with any model on Hugging Face"

4. Run a demonstration with a calculation task:
   - "Let's ask: What is 345 multiplied by 22?"
   - Show how the agent breaks down the problem
   - Point out the reasoning steps in the output
   - "Notice how it correctly identifies it should use the Calculator tool"

## Demo 2: Plan-and-Execute Pattern (5 minutes)

1. Create a new flow with the Plan-Execute Agent:
   - Drag in the "Plan-Execute Agent" component 
   - Connect the same HuggingFace model
   - Connect tools (calculator, search, etc.)
   - Connect chat input/output

2. Explain the difference:
   - "This agent first creates a plan with discrete steps"
   - "Then it executes each step individually using appropriate tools"
   - "This is especially powerful for complex, multi-step tasks"

3. Run a demonstration with a complex query:
   - "Let's ask: If I invest $1000 with 5% annual interest compounded monthly, how much will I have after 3 years?"
   - Show how it first plans the approach
   - Then executes each step
   - "Notice how it breaks down the problem into manageable parts"

## Technical Implementation (3 minutes)

1. Show the code structure:
   ```
   custom_components/
   └── hf_native_agents/
       ├── __init__.py
       ├── react_agent.py
       ├── plan_execute_agent.py
       ├── tests/
       └── sample_flows/
   ```

2. Highlight the key technical aspects:
   - "These components use LangChain's agent frameworks but don't require tool-calling APIs"
   - "They work with any standard LangChain tool without modification"
   - "We've added comprehensive tests to ensure reliability"

## Benefits and Conclusion (2 minutes)

"To summarize the benefits:

1. **Open-source compatibility**: Works with Mistral, Llama, and other open models
2. **Cost efficiency**: No dependency on expensive API calls
3. **Full tool support**: All the same capabilities as OpenAI agents
4. **Flexibility**: Two different agent patterns for different use cases
5. **Easy integration**: Drag-and-drop simplicity in the Langflow UI

This implementation fully meets the requirements we discussed - enabling the use of open-source LLMs with agent capabilities and supporting all standard LangChain tools."

## Q&A (As needed)

Be prepared to answer questions about:
- Specific model performance
- Support for additional tools
- Performance considerations
- How to customize the agent prompts
- Integration with existing Langflow projects
