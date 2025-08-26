# docker/Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONPATH=/workspace:$PYTHONPATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    chromium-browser \
    chromium-driver \
    firefox \
    firefox-geckodriver \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python package managers
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create workspace
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements-docker.txt /workspace/
RUN pip3 install --no-cache-dir -r requirements-docker.txt

# Install PyTorch with CUDA support
RUN pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install WebArena environment
RUN git clone https://github.com/web-arena-x/webarena.git /tmp/webarena && \
    cd /tmp/webarena && \
    pip3 install -e . && \
    cd / && rm -rf /tmp/webarena/.git

# Install additional ML libraries
RUN pip3 install \
    transformers==4.35.0 \
    accelerate==0.24.1 \
    datasets==2.14.6 \
    wandb==0.16.0 \
    tensorboard==2.15.1 \
    ray[tune]==2.8.0

# Install API dependencies
RUN pip3 install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    redis==5.0.1 \
    celery==5.3.4

# Install LLM provider SDKs
RUN pip3 install \
    openai==1.3.5 \
    anthropic==0.7.7 \
    google-generativeai==0.3.0

# Copy application code
COPY mas_webarena /workspace/mas_webarena
COPY scripts /workspace/scripts
COPY training /workspace/training
COPY api /workspace/api
COPY config /workspace/config

# Make scripts executable
RUN chmod +x /workspace/scripts/*.sh

# Create necessary directories
RUN mkdir -p /workspace/models \
             /workspace/data \
             /workspace/logs \
             /workspace/checkpoints \
             /workspace/dag_cache \
             /workspace/results

# Expose ports
EXPOSE 8000 8001 6006 8888 8265

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["/workspace/scripts/entrypoint.sh"]
CMD ["train"]