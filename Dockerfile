FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv for better Python package management
RUN curl -L --output /tmp/uv-installer.sh https://astral.sh/uv/install.sh \
    && chmod +x /tmp/uv-installer.sh \
    && /tmp/uv-installer.sh \
    && rm /tmp/uv-installer.sh \
    && echo 'PATH="/root/.cargo/bin:$PATH"' >> ~/.bashrc

# Copy requirements and setup files first for better caching
COPY requirements-core.txt .
COPY requirements-dev.txt .
COPY setup.py .
COPY README.md .

# Copy source code
COPY src/ src/
COPY langflow_extensions/ langflow_extensions/
COPY tests/ tests/
COPY pytest.ini .

# Create separate virtual environments to avoid conflicts
# 1. Core environment (for standalone functionality)
RUN . ~/.bashrc && python -m venv /opt/venv-core
RUN . ~/.bashrc && . /opt/venv-core/bin/activate && \
    uv pip install -r requirements-core.txt && \
    uv pip install -e .

# --- Langflow Dependencies Stage (optional) ---
ARG INSTALL_LANGFLOW
# Install Langflow dependencies INTO the core venv if requested
RUN if [ "$INSTALL_LANGFLOW" = "true" ] ; then \
        . ~/.bashrc && \
        . /opt/venv-core/bin/activate && \
        echo "Installing Langflow/Dev dependencies into venv-core..." && \
        uv pip install -r requirements-dev.txt && \
        echo "Langflow/Dev dependencies installed into venv-core." ; \
    fi

# Install our custom Langflow extensions package when Langflow is installed
RUN if [ "$INSTALL_LANGFLOW" = "true" ] ; then \
        . ~/.bashrc && \
        . /opt/venv-core/bin/activate && \
        echo "Installing custom Langflow extensions..." && \
        cd /app/langflow_extensions && \
        pip install -e . && \
        echo "Custom Langflow extensions installed." ; \
    fi

# Activation script for the single environment
RUN echo '#!/bin/bash\n. /opt/venv-core/bin/activate' > /activate-environment.sh && chmod +x /activate-environment.sh
# Remove old activation scripts if they exist (cleanup)
RUN rm -f /activate-core.sh /activate-langflow.sh

# Default command runs pytest
CMD /activate-environment.sh && pytest tests/unit
