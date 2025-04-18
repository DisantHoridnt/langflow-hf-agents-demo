#!/bin/bash

# HF-Native Agents Demo Launcher
# This script sets up and launches the Langflow demo with HF-Native agent components

echo "🚀 Starting HF-Native Agents Demo"
echo "---------------------------------"

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Set environment variable for custom components
echo "🔧 Setting LANGFLOW_COMPONENT_PATH..."
export LANGFLOW_COMPONENT_PATH=$PWD/custom_components

# Check if HF API token is set
if [ -z "$HUGGINGFACEHUB_API_TOKEN" ]; then
  echo "⚠️  Warning: HUGGINGFACEHUB_API_TOKEN environment variable is not set"
  echo "   You'll need to enter your API token in the Langflow UI"
  echo "   To set it permanently, run: export HUGGINGFACEHUB_API_TOKEN=your_token_here"
  echo ""
fi

# Print demo info
echo "ℹ️  Demo Information:"
echo "   - UI will be available at http://localhost:7860"
echo "   - Sample flows are in custom_components/hf_native_agents/sample_flows/"
echo "   - Demo script is available in demo_script.md"
echo ""

# Start Langflow
echo "▶️  Starting Langflow..."
langflow run
