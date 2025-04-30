"""Setup script for the langflow_extensions package."""

from setuptools import setup, find_packages

setup(
    name="langflow_extensions",
    version="0.1.0",
    description="Custom components for Langflow",
    author="Langflow HF Agents Team",
    packages=find_packages(),
    install_requires=[
        "langflow",
        "langchain",
    ],
    entry_points={
        "langflow.components": [
            "custom_agents=langflow_extensions.custom_agents",
        ],
    },
    python_requires=">=3.8",
)
