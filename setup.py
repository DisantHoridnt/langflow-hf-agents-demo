"""
Setup script for installing the hf-agents package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Load core requirements
with open("requirements-core.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if not line.startswith("#")]

# Load dev requirements
with open("requirements-dev.txt", "r", encoding="utf-8") as f:
    dev_requirements = [line.strip() for line in f if not line.startswith("#")]

setup(
    name="hf-agents",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Hugging Face Native Agent Components for Langflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hf-agents",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "langflow": ["langflow>=0.5.0"],
    },
)
