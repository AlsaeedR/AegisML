# Hardened Dockerfile for AegisML Agent 2 Sandbox Worker
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install security and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install core ML, security, and testing libraries
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    joblib \
    nltk \
    matplotlib \
    seaborn \
    tqdm \
    adversarial-robustness-toolbox==1.20.1 \
    pydantic

# Pre-download NLTK stopwords and punkt to avoid runtime network calls
RUN python -m nltk.downloader stopwords punkt

# Create unprivileged aegis user and group
RUN groupadd -g 1000 aegis && \
    useradd -u 1000 -g aegis -s /bin/bash -m aegis

# Create workspace mount points
RUN mkdir -p /workspace/data /workspace/input /workspace/output && \
    chown -R aegis:aegis /workspace /app

# Copy application code into container
COPY --chown=aegis:aegis src/ /app/src/

USER aegis

# Entrypoint executes the sandbox worker
ENTRYPOINT ["python", "-m", "src.agents.testing_agent.sandbox_worker"]

