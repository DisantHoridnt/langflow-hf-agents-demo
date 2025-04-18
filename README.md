# HF-Native Agents for Langflow - Client Demo

This is a demonstration of open-source LLM agents for Langflow that work with Hugging Face models without requiring OpenAI or other proprietary APIs. The agents support both ReAct and Plan-and-Execute patterns for solving complex reasoning tasks.

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
