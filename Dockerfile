# -----------------------------------------------------------
# Dockerfile for Homomorphic Encryption in Federated Learning
# Bachelor Thesis Reproducibility Environment
# -----------------------------------------------------------

FROM python:3.10-slim

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for PyTorch, TenSEAL (C++/cmake), and OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy all project files (scripts, README, etc.)
COPY . .

# Default output folder inside container
RUN mkdir -p /workspace/runs

# Entry point can be overridden at docker run
ENTRYPOINT ["python3"]
