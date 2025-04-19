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
COPY tests/ tests/
COPY pytest.ini .

# Create separate virtual environments to avoid conflicts
# 1. Core environment (for standalone functionality)
RUN . ~/.bashrc && python -m venv /opt/venv-core
RUN . ~/.bashrc && . /opt/venv-core/bin/activate && \
    uv pip install -r requirements-core.txt && \
    uv pip install -e . && \
    uv pip install -r requirements-dev.txt

# 2. Langflow environment (only if needed)
ARG INSTALL_LANGFLOW=false
RUN if [ "$INSTALL_LANGFLOW" = "true" ] ; then \
        . ~/.bashrc && python -m venv /opt/venv-langflow && \
        . /opt/venv-langflow/bin/activate && \
        uv pip install -e . && \
        uv pip install langflow ; \
    fi

# Create activation scripts for each environment
RUN echo '#!/bin/bash\n. /opt/venv-core/bin/activate' > /activate-core.sh && chmod +x /activate-core.sh
RUN if [ "$INSTALL_LANGFLOW" = "true" ] ; then \
        echo '#!/bin/bash\n. /opt/venv-langflow/bin/activate' > /activate-langflow.sh && chmod +x /activate-langflow.sh ; \
    fi

# Default command runs pytest
CMD ["python", "-m", "pytest"]
