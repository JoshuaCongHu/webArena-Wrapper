# CLAUDE_DOCKER_DEPLOYMENT.md - Complete Docker & Training Infrastructure Implementation

## 🎯 Implementation Overview

This document provides complete implementation instructions for Docker containerization, RunPod deployment, and distributed training infrastructure for the WebArena MAS LLM orchestrator. Follow each section sequentially to achieve a production-ready deployment.

---

## 📁 File Structure to Create

```
mas_webarena/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── scripts/
│   ├── entrypoint.sh
│   ├── train_distributed.py
│   ├── deploy_runpod.py
│   ├── monitor_training.py
│   └── evaluate.py
├── training/
│   ├── __init__.py
│   ├── distributed_trainer.py
│   ├── data_loader.py
│   └── checkpoint_manager.py
├── api/
│   ├── __init__.py
│   ├── server.py
│   └── models.py
├── config/
│   ├── test_training.yaml
│   ├── standard_training.yaml
│   └── full_training.yaml
├── deployment/
│   ├── runpod-template.json
│   ├── kubernetes.yaml
│   └── nginx.conf
├── .env.example
├── requirements-docker.txt
├── QUICKSTART.md
└── Makefile
```

---

## 🐳 Section 1: Docker Configuration

### 1.1 Create Dockerfile

```dockerfile
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
```

### 1.2 Create docker-compose.yml

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  mas-trainer:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: ${DOCKER_REGISTRY:-local}/webarena-mas:${VERSION:-latest}
    container_name: mas-trainer
    hostname: mas-trainer
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - WANDB_API_KEY=${WANDB_API_KEY}
      - WANDB_PROJECT=webarena-mas
      - CUDA_LAUNCH_BLOCKING=1
      - TRAINING_MODE=distributed
      - REDIS_URL=redis://redis:6379
    volumes:
      - ../data:/workspace/data
      - ../models:/workspace/models
      - ../logs:/workspace/logs
      - ../results:/workspace/results
      - ../checkpoints:/workspace/checkpoints
      - model-cache:/root/.cache
      - dag-cache:/workspace/dag_cache
    ports:
      - "6006:6006"  # TensorBoard
      - "8888:8888"  # Jupyter
    networks:
      - mas-network
    shm_size: '32gb'
    ulimits:
      memlock:
        soft: -1
        hard: -1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: ["train", "--method", "${METHOD:-p3o}", "--distributed"]
    depends_on:
      - redis

  mas-api:
    image: ${DOCKER_REGISTRY:-local}/webarena-mas:${VERSION:-latest}
    container_name: mas-api
    hostname: mas-api
    runtime: nvidia
    environment:
      - MODEL_PATH=/workspace/models/best_checkpoint.pt
      - REDIS_URL=redis://redis:6379
      - API_WORKERS=4
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ../models:/workspace/models:ro
      - ../logs:/workspace/logs
      - dag-cache:/workspace/dag_cache
    ports:
      - "8000:8000"  # API
    networks:
      - mas-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: ["serve", "--port", "8000", "--workers", "4"]
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: mas-redis
    hostname: redis
    ports:
      - "6379:6379"
    networks:
      - mas-network
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  monitoring:
    image: grafana/grafana:latest
    container_name: mas-monitoring
    hostname: monitoring
    ports:
      - "3000:3000"
    networks:
      - mas-network
    volumes:
      - grafana-data:/var/lib/grafana
      - ../deployment/grafana:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=redis-datasource
    restart: unless-stopped

networks:
  mas-network:
    driver: bridge

volumes:
  model-cache:
  dag-cache:
  redis-data:
  grafana-data:
```

### 1.3 Create .dockerignore

```
# docker/.dockerignore
*.pyc
__pycache__
*.pyo
*.pyd
.Python
env/
venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.gitignore
.mypy_cache
.pytest_cache
.hypothesis
*.egg-info/
dist/
build/
*.egg
*.whl
.DS_Store
.idea/
.vscode/
*.swp
*.swo
*~
.env
.env.*
!.env.example
data/
models/
logs/
results/
checkpoints/
*.pt
*.pth
*.ckpt
wandb/
runs/
```

---

## 📜 Section 2: Scripts Implementation

### 2.1 Create entrypoint.sh

```bash
#!/bin/bash
# scripts/entrypoint.sh

set -e

# Function to wait for services
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    echo "Waiting for $service at $host:$port..."
    while ! nc -z $host $port; do
        sleep 1
    done
    echo "$service is ready!"
}

# Setup display for headless browser
if [ "$ENABLE_BROWSER" = "true" ]; then
    Xvfb :99 -screen 0 1920x1080x24 &
    export DISPLAY=:99
fi

# Wait for Redis if needed
if [ ! -z "$REDIS_URL" ]; then
    REDIS_HOST=$(echo $REDIS_URL | sed -E 's/redis:\/\/([^:]+).*/\1/')
    wait_for_service ${REDIS_HOST:-redis} 6379 "Redis"
fi

# Parse command
COMMAND=${1:-train}
shift

case $COMMAND in
    train)
        echo "Starting distributed training..."
        exec python3 /workspace/scripts/train_distributed.py "$@"
        ;;
    
    serve)
        echo "Starting API server..."
        exec python3 /workspace/api/server.py "$@"
        ;;
    
    evaluate)
        echo "Running evaluation..."
        exec python3 /workspace/scripts/evaluate.py "$@"
        ;;
    
    jupyter)
        echo "Starting Jupyter notebook..."
        exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
        ;;
    
    tensorboard)
        echo "Starting TensorBoard..."
        exec tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006
        ;;
    
    worker)
        echo "Starting Celery worker..."
        exec celery -A training.tasks worker --loglevel=info "$@"
        ;;
    
    bash)
        exec /bin/bash "$@"
        ;;
    
    test)
        echo "Running tests..."
        exec python3 -m pytest /workspace/tests "$@"
        ;;
    
    *)
        echo "Unknown command: $COMMAND"
        echo "Available commands: train, serve, evaluate, jupyter, tensorboard, worker, bash, test"
        exit 1
        ;;
esac
```

### 2.2 Create train_distributed.py

```python
# scripts/train_distributed.py
#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Add workspace to path
sys.path.append('/workspace')
sys.path.append('/workspace/mas_webarena')

import wandb
from training.distributed_trainer import DistributedTrainer
from training.data_loader import WebArenaDataLoader
from training.checkpoint_manager import CheckpointManager
from mas.enhanced_webarena_mas import EnhancedWebArenaMAS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    
    # Set CUDA device
    torch.cuda.set_device(rank)
    logger.info(f"Initialized rank {rank}/{world_size} on GPU {rank}")

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_rank(rank: int, world_size: int, args):
    """Training function for each GPU/process"""
    try:
        # Setup distributed if multi-GPU
        if world_size > 1:
            setup_distributed(rank, world_size)
        
        # Initialize wandb on main process
        if rank == 0 and args.use_wandb:
            wandb.init(
                project=os.environ.get('WANDB_PROJECT', 'webarena-mas'),
                name=f"{args.method}-{args.run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=vars(args),
                tags=[args.method, f"gpu-{world_size}"]
            )
        
        # Create MAS instance
        logger.info(f"Creating MAS instance on rank {rank}")
        mas = EnhancedWebArenaMAS(
            method=args.method,
            budget=args.budget,
            use_llm_orchestrator=args.use_llm_orchestrator,
            llm_model=args.llm_model,
            enable_replanning=args.enable_replanning,
            num_agents=args.num_agents,
            max_nodes=args.max_nodes,
            device=f'cuda:{rank}'
        )
        
        # Wrap model for DDP if multi-GPU
        if world_size > 1:
            if hasattr(mas.algorithm, 'policy_net'):
                mas.algorithm.policy_net = DDP(
                    mas.algorithm.policy_net.to(rank),
                    device_ids=[rank],
                    output_device=rank,
                    find_unused_parameters=True
                )
                logger.info(f"Wrapped policy network in DDP on rank {rank}")
            
            if hasattr(mas.algorithm, 'critic_net'):
                mas.algorithm.critic_net = DDP(
                    mas.algorithm.critic_net.to(rank),
                    device_ids=[rank],
                    output_device=rank,
                    find_unused_parameters=True
                )
        
        # Create data loader
        data_loader = WebArenaDataLoader(
            data_path=args.data_path,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=args.checkpoint_dir,
            rank=rank,
            keep_last_n=args.keep_checkpoints
        )
        
        # Create trainer
        trainer = DistributedTrainer(
            mas=mas,
            data_loader=data_loader,
            checkpoint_manager=checkpoint_manager,
            rank=rank,
            world_size=world_size,
            args=args
        )
        
        # Resume from checkpoint if specified
        start_episode = 0
        if args.resume:
            checkpoint = checkpoint_manager.load_checkpoint(args.resume)
            if checkpoint:
                start_episode = checkpoint.get('episode', 0) + 1
                logger.info(f"Resumed from episode {start_episode}")
        
        # Training loop
        logger.info(f"Starting training from episode {start_episode}")
        trainer.train(start_episode=start_episode)
        
        # Cleanup
        if world_size > 1:
            cleanup_distributed()
        
        if rank == 0 and args.use_wandb:
            wandb.finish()
            
    except Exception as e:
        logger.error(f"Training failed on rank {rank}: {e}")
        raise

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Distributed training for WebArena MAS')
    
    # Model configuration
    parser.add_argument('--method', type=str, default='p3o',
                       choices=['p3o', 'ppo_lagrangian', 'macpo'],
                       help='RL method to use')
    parser.add_argument('--budget', type=float, default=1.0,
                       help='Budget constraint')
    parser.add_argument('--num-agents', type=int, default=4,
                       help='Number of agents in the system')
    parser.add_argument('--max-nodes', type=int, default=10,
                       help='Maximum nodes in DAG')
    
    # LLM configuration
    parser.add_argument('--use-llm-orchestrator', action='store_true', default=True,
                       help='Use LLM orchestrator')
    parser.add_argument('--no-llm-orchestrator', dest='use_llm_orchestrator', 
                       action='store_false')
    parser.add_argument('--llm-model', type=str, default='gpt-4-turbo',
                       choices=['gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo',
                               'claude-3-opus', 'claude-3-sonnet', 
                               'gemini-1.5-pro'],
                       help='LLM model to use')
    parser.add_argument('--enable-replanning', action='store_true', default=True,
                       help='Enable dynamic replanning')
    
    # Training configuration
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--ppo-epochs', type=int, default=4,
                       help='PPO epochs per update')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true', default=False,
                       help='Enable distributed training')
    parser.add_argument('--num-gpus', type=int, default=None,
                       help='Number of GPUs to use (default: all available)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Data configuration
    parser.add_argument('--data-path', type=str, default='/workspace/data',
                       help='Path to training data')
    parser.add_argument('--eval-interval', type=int, default=100,
                       help='Evaluation interval')
    parser.add_argument('--save-interval', type=int, default=500,
                       help='Checkpoint save interval')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='/workspace/checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--keep-checkpoints', type=int, default=5,
                       help='Number of checkpoints to keep')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='/workspace/logs',
                       help='Log directory')
    parser.add_argument('--use-wandb', action='store_true', default=True,
                       help='Use Weights & Biases logging')
    parser.add_argument('--run-name', type=str, default='experiment',
                       help='Name for this run')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Determine number of GPUs
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
    
    logger.info(f"Training configuration: {vars(args)}")
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_path).mkdir(parents=True, exist_ok=True)
    
    # Launch training
    if args.distributed and args.num_gpus > 1:
        logger.info(f"Starting distributed training on {args.num_gpus} GPUs")
        mp.spawn(
            train_rank,
            args=(args.num_gpus, args),
            nprocs=args.num_gpus,
            join=True
        )
    else:
        logger.info("Starting single-GPU training")
        train_rank(0, 1, args)

if __name__ == "__main__":
    main()
```

### 2.3 Create deploy_runpod.py

```python
# scripts/deploy_runpod.py
#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import requests
import runpod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunPodDeployment:
    """Manage RunPod deployments for training"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        runpod.api_key = api_key
        
    def create_training_pod(self, config: Dict[str, Any]) -> str:
        """Create a new training pod on RunPod"""
        
        # Build pod configuration
        pod_config = {
            "name": f"webarena-mas-{config['method']}-{int(time.time())}",
            "image": config.get("image", "webarena-mas:latest"),
            "gpu_type_id": self._get_gpu_type_id(config.get("gpu_type", "NVIDIA RTX A6000")),
            "gpu_count": config.get("gpu_count", 1),
            "volume_in_gb": config.get("volume_gb", 100),
            "container_disk_in_gb": 50,
            "min_memory_in_gb": config.get("memory_gb", 32),
            "docker_args": self._build_docker_command(config),
            "env": self._build_environment(config),
            "ports": "8000/http,6006/http,8888/http,22/tcp",
            "volume_mount_path": "/workspace/persistent",
            "start_jupyter": config.get("start_jupyter", False),
            "start_ssh": config.get("start_ssh", True)
        }
        
        logger.info(f"Creating pod with configuration: {json.dumps(pod_config, indent=2)}")
        
        try:
            response = runpod.create_pod(**pod_config)
            pod_id = response["id"]
            logger.info(f"Successfully created pod: {pod_id}")
            
            # Wait for pod to be ready
            self._wait_for_pod_ready(pod_id)
            
            return pod_id
            
        except Exception as e:
            logger.error(f"Failed to create pod: {e}")
            raise
    
    def _get_gpu_type_id(self, gpu_type: str) -> str:
        """Map GPU type name to RunPod GPU type ID"""
        gpu_mapping = {
            "NVIDIA RTX A6000": "NVIDIA RTX A6000",
            "NVIDIA RTX A5000": "NVIDIA RTX A5000",
            "NVIDIA RTX 3090": "NVIDIA GeForce RTX 3090",
            "NVIDIA A100": "NVIDIA A100 80GB PCIe",
            "NVIDIA A100-40GB": "NVIDIA A100-PCIE-40GB",
            "NVIDIA A40": "NVIDIA A40",
            "NVIDIA RTX 4090": "NVIDIA GeForce RTX 4090"
        }
        return gpu_mapping.get(gpu_type, gpu_type)
    
    def _build_docker_command(self, config: Dict[str, Any]) -> str:
        """Build Docker command for training"""
        cmd_parts = [
            "train",
            "--method", config.get("method", "p3o"),
            "--episodes", str(config.get("episodes", 10000)),
            "--batch-size", str(config.get("batch_size", 32)),
            "--num-gpus", str(config.get("gpu_count", 1)),
            "--checkpoint-dir", "/workspace/persistent/checkpoints",
            "--data-path", "/workspace/persistent/data",
            "--log-dir", "/workspace/persistent/logs"
        ]
        
        if config.get("distributed", True) and config.get("gpu_count", 1) > 1:
            cmd_parts.append("--distributed")
        
        if config.get("use_llm_orchestrator", True):
            cmd_parts.extend([
                "--use-llm-orchestrator",
                "--llm-model", config.get("llm_model", "gpt-4-turbo")
            ])
        
        if config.get("enable_replanning", True):
            cmd_parts.append("--enable-replanning")
        
        if config.get("use_wandb", True):
            cmd_parts.extend([
                "--use-wandb",
                "--run-name", config.get("run_name", "runpod-experiment")
            ])
        
        if config.get("resume"):
            cmd_parts.extend(["--resume", config["resume"]])
        
        return " ".join(cmd_parts)
    
    def _build_environment(self, config: Dict[str, Any]) -> list:
        """Build environment variables for pod"""
        env_vars = []
        
        # API Keys
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "WANDB_API_KEY"]:
            value = config.get(key) or os.environ.get(key)
            if value:
                env_vars.append({"key": key, "value": value})
        
        # Training configuration
        env_vars.extend([
            {"key": "TRAINING_MODE", "value": "distributed" if config.get("distributed", True) else "single"},
            {"key": "CUDA_VISIBLE_DEVICES", "value": ",".join(map(str, range(config.get("gpu_count", 1))))},
            {"key": "WANDB_PROJECT", "value": config.get("wandb_project", "webarena-mas")},
            {"key": "PYTHONPATH", "value": "/workspace:/workspace/mas_webarena"}
        ])
        
        return env_vars
    
    def _wait_for_pod_ready(self, pod_id: str, timeout: int = 300):
        """Wait for pod to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            pod = runpod.get_pod(pod_id)
            
            if pod["status"] == "RUNNING":
                logger.info(f"Pod {pod_id} is ready")
                return
            elif pod["status"] == "FAILED":
                raise RuntimeError(f"Pod {pod_id} failed to start")
            
            logger.info(f"Waiting for pod to be ready... Status: {pod['status']}")
            time.sleep(10)
        
        raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout} seconds")
    
    def get_pod_status(self, pod_id: str) -> Dict[str, Any]:
        """Get current status of a pod"""
        pod = runpod.get_pod(pod_id)
        return {
            "id": pod_id,
            "status": pod["status"],
            "gpu_type": pod.get("machine", {}).get("gpu_type"),
            "gpu_count": pod.get("machine", {}).get("gpu_count"),
            "runtime": pod.get("runtime"),
            "cost_per_hour": pod.get("cost_per_hour")
        }
    
    def stream_logs(self, pod_id: str):
        """Stream logs from a running pod"""
        logger.info(f"Streaming logs from pod {pod_id}")
        
        for log in runpod.stream_logs(pod_id):
            print(log.strip())
    
    def stop_pod(self, pod_id: str):
        """Stop a running pod"""
        logger.info(f"Stopping pod {pod_id}")
        runpod.stop_pod(pod_id)
    
    def terminate_pod(self, pod_id: str):
        """Terminate a pod"""
        logger.info(f"Terminating pod {pod_id}")
        runpod.terminate_pod(pod_id)
    
    def list_pods(self) -> list:
        """List all pods"""
        pods = runpod.get_pods()
        return [{
            "id": pod["id"],
            "name": pod["name"],
            "status": pod["status"],
            "gpu_type": pod.get("machine", {}).get("gpu_type"),
            "created": pod.get("created_at")
        } for pod in pods]

def load_training_config(config_file: str) -> Dict[str, Any]:
    """Load training configuration from file"""
    config_path = Path(config_file)
    
    if config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    elif config_path.suffix in ['.yaml', '.yml']:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")

def main():
    parser = argparse.ArgumentParser(description='Deploy training to RunPod')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create pod command
    create_parser = subparsers.add_parser('create', help='Create a new training pod')
    create_parser.add_argument('--config', type=str, required=True,
                             help='Configuration file (JSON or YAML)')
    create_parser.add_argument('--api-key', type=str, 
                             default=os.environ.get('RUNPOD_API_KEY'),
                             help='RunPod API key')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get pod status')
    status_parser.add_argument('pod_id', type=str, help='Pod ID')
    status_parser.add_argument('--api-key', type=str,
                             default=os.environ.get('RUNPOD_API_KEY'))
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Stream pod logs')
    logs_parser.add_argument('pod_id', type=str, help='Pod ID')
    logs_parser.add_argument('--api-key', type=str,
                           default=os.environ.get('RUNPOD_API_KEY'))
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop a pod')
    stop_parser.add_argument('pod_id', type=str, help='Pod ID')
    stop_parser.add_argument('--api-key', type=str,
                           default=os.environ.get('RUNPOD_API_KEY'))
    
    # Terminate command
    terminate_parser = subparsers.add_parser('terminate', help='Terminate a pod')
    terminate_parser.add_argument('pod_id', type=str, help='Pod ID')
    terminate_parser.add_argument('--api-key', type=str,
                                default=os.environ.get('RUNPOD_API_KEY'))
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all pods')
    list_parser.add_argument('--api-key', type=str,
                           default=os.environ.get('RUNPOD_API_KEY'))
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if not args.api_key:
        logger.error("RunPod API key is required. Set RUNPOD_API_KEY environment variable or use --api-key")
        return
    
    deployment = RunPodDeployment(args.api_key)
    
    if args.command == 'create':
        config = load_training_config(args.config)
        pod_id = deployment.create_training_pod(config)
        print(f"Created pod: {pod_id}")
        
        # Optionally stream logs
        if config.get("stream_logs", False):
            deployment.stream_logs(pod_id)
    
    elif args.command == 'status':
        status = deployment.get_pod_status(args.pod_id)
        print(json.dumps(status, indent=2))
    
    elif args.command == 'logs':
        deployment.stream_logs(args.pod_id)
    
    elif args.command == 'stop':
        deployment.stop_pod(args.pod_id)
        print(f"Stopped pod: {args.pod_id}")
    
    elif args.command == 'terminate':
        deployment.terminate_pod(args.pod_id)
        print(f"Terminated pod: {args.pod_id}")
    
    elif args.command == 'list':
        pods = deployment.list_pods()
        for pod in pods:
            print(f"{pod['id']}: {pod['name']} ({pod['status']}) - {pod['gpu_type']}")

if __name__ == "__main__":
    main()
```

---

## 🎓 Section 3: Training Infrastructure

### 3.1 Create distributed_trainer.py

```python
# training/distributed_trainer.py
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from typing import Dict, Any, List, Optional
import time
import logging
from pathlib import Path
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class DistributedTrainer:
    """Distributed trainer for WebArena MAS"""
    
    def __init__(
        self,
        mas,
        data_loader,
        checkpoint_manager,
        rank: int,
        world_size: int,
        args
    ):
        self.mas = mas
        self.data_loader = data_loader
        self.checkpoint_manager = checkpoint_manager
        self.rank = rank
        self.world_size = world_size
        self.args = args
        
        # Training state
        self.episode = 0
        self.global_step = 0
        self.best_success_rate = 0.0
        self.best_cost_guarantee_rate = 0.0
        
        # Metrics tracking
        self.episode_metrics = defaultdict(list)
        self.validation_metrics = defaultdict(list)
        
        # Setup tensorboard
        if self.rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=f"{args.log_dir}/tensorboard")
    
    def train(self, start_episode: int = 0):
        """Main training loop"""
        self.episode = start_episode
        
        logger.info(f"Starting training from episode {start_episode}")
        
        for episode in range(start_episode, self.args.episodes):
            self.episode = episode
            
            # Set epoch for data sampler
            if hasattr(self.data_loader, 'sampler') and hasattr(self.data_loader.sampler, 'set_epoch'):
                self.data_loader.sampler.set_epoch(episode)
            
            # Training epoch
            epoch_metrics = self._train_epoch()
            
            # Synchronize metrics across GPUs
            if self.world_size > 1:
                epoch_metrics = self._aggregate_metrics(epoch_metrics)
            
            # Logging and checkpointing on main process
            if self.rank == 0:
                # Log metrics
                self._log_metrics(epoch_metrics, prefix='train')
                
                # Evaluation
                if episode % self.args.eval_interval == 0:
                    eval_metrics = self._evaluate()
                    self._log_metrics(eval_metrics, prefix='eval')
                    
                    # Save best model
                    if eval_metrics['success_rate'] > self.best_success_rate:
                        self.best_success_rate = eval_metrics['success_rate']
                        self.checkpoint_manager.save_checkpoint(
                            self.mas,
                            episode,
                            eval_metrics,
                            is_best=True
                        )
                        logger.info(f"New best model: success_rate={self.best_success_rate:.3f}")
                
                # Regular checkpointing
                if episode % self.args.save_interval == 0:
                    self.checkpoint_manager.save_checkpoint(
                        self.mas,
                        episode,
                        epoch_metrics,
                        is_best=False
                    )
            
            # Synchronize before next epoch
            if self.world_size > 1:
                dist.barrier()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_metrics = defaultdict(list)
        
        for batch_idx, batch_tasks in enumerate(self.data_loader):
            batch_start_time = time.time()
            
            # Process batch
            batch_results = []
            for task in batch_tasks:
                # Execute task with MAS
                result = self.mas.solve_task(task)
                batch_results.append(result)
                
                # Update metrics
                epoch_metrics['success'].append(float(result['success']))
                epoch_metrics['cost'].append(result['cost'])
                epoch_metrics['reward'].append(result['reward'])
                epoch_metrics['cgr'].append(float(result['cost'] <= self.args.budget * 1.05))
                
                if 'replanning_count' in result:
                    epoch_metrics['replanning_count'].append(result['replanning_count'])
                
                # Track DAG metrics
                if 'dag_metrics' in result:
                    for key, value in result['dag_metrics'].items():
                        epoch_metrics[f'dag_{key}'].append(value)
            
            # Log batch progress
            batch_time = time.time() - batch_start_time
            if self.rank == 0 and batch_idx % 10 == 0:
                logger.info(
                    f"Episode {self.episode}, Batch {batch_idx}/{len(self.data_loader)}: "
                    f"Success={np.mean(epoch_metrics['success'][-len(batch_results):])*100:.1f}%, "
                    f"Cost={np.mean(epoch_metrics['cost'][-len(batch_results):]):.3f}, "
                    f"Time={batch_time:.2f}s"
                )
            
            self.global_step += 1
        
        # Compute epoch averages
        return {
            key: np.mean(values) if values else 0.0
            for key, values in epoch_metrics.items()
        }
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        logger.info("Running evaluation...")
        
        eval_metrics = defaultdict(list)
        
        # Switch to evaluation mode if applicable
        if hasattr(self.mas.algorithm, 'eval'):
            self.mas.algorithm.eval()
        
        # Run evaluation tasks
        with torch.no_grad():
            for task in self.data_loader.get_eval_tasks(num_tasks=50):
                result = self.mas.solve_task(task)
                
                eval_metrics['success'].append(float(result['success']))
                eval_metrics['cost'].append(result['cost'])
                eval_metrics['reward'].append(result['reward'])
                eval_metrics['cgr'].append(float(result['cost'] <= self.args.budget * 1.05))
        
        # Switch back to training mode
        if hasattr(self.mas.algorithm, 'train'):
            self.mas.algorithm.train()
        
        # Return averages
        return {
            'success_rate': np.mean(eval_metrics['success']),
            'avg_cost': np.mean(eval_metrics['cost']),
            'avg_reward': np.mean(eval_metrics['reward']),
            'cost_guarantee_rate': np.mean(eval_metrics['cgr'])
        }
    
    def _aggregate_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Aggregate metrics across all processes"""
        aggregated = {}
        
        for key, value in metrics.items():
            # Convert to tensor
            tensor = torch.tensor(value, dtype=torch.float32, device=f'cuda:{self.rank}')
            
            # All-reduce sum
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            
            # Average across processes
            aggregated[key] = tensor.item() / self.world_size
        
        return aggregated
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = 'train'):
        """Log metrics to tensorboard and wandb"""
        # TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{key}', value, self.episode)
        
        # Weights & Biases
        if self.args.use_wandb:
            import wandb
            wandb_metrics = {f'{prefix}/{key}': value for key, value in metrics.items()}
            wandb_metrics['episode'] = self.episode
            wandb.log(wandb_metrics)
        
        # Console logging
        logger.info(
            f"Episode {self.episode} [{prefix}]: "
            f"Success={metrics.get('success', 0)*100:.1f}%, "
            f"Cost={metrics.get('cost', 0):.3f}, "
            f"CGR={metrics.get('cgr', 0)*100:.1f}%"
        )
```

### 3.2 Create data_loader.py

```python
# training/data_loader.py
import json
import random
from pathlib import Path
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import logging

logger = logging.getLogger(__name__)

class WebArenaDataset(Dataset):
    """WebArena task dataset"""
    
    def __init__(self, data_path: str, split: str = 'train'):
        self.data_path = Path(data_path)
        self.split = split
        self.tasks = self._load_tasks()
        
    def _load_tasks(self) -> List[Dict]:
        """Load tasks from files or generate synthetic tasks"""
        task_file = self.data_path / f'{self.split}_tasks.json'
        
        if task_file.exists():
            logger.info(f"Loading tasks from {task_file}")
            with open(task_file, 'r') as f:
                return json.load(f)
        else:
            logger.info("Generating synthetic tasks")
            return self._generate_synthetic_tasks()
    
    def _generate_synthetic_tasks(self, num_tasks: int = 10000) -> List[Dict]:
        """Generate synthetic WebArena tasks"""
        task_templates = [
            {
                'intent': 'Navigate to {site} and search for {item}',
                'sites': ['shopping.com', 'amazon.com', 'ebay.com'],
                'difficulty': 'easy',
                'expected_steps': 3
            },
            {
                'intent': 'Login to {site} and update profile information',
                'sites': ['reddit.com', 'gitlab.com', 'maps.google.com'],
                'difficulty': 'medium',
                'expected_steps': 5
            },
            {
                'intent': 'Book a {service} on {site} for {date}',
                'sites': ['classifieds.com', 'shopping.com'],
                'difficulty': 'hard',
                'expected_steps': 7
            },
            {
                'intent': 'Complete checkout process for items in cart',
                'sites': ['shopping.com'],
                'difficulty': 'hard',
                'expected_steps': 8
            }
        ]
        
        items = ['laptop', 'book', 'phone', 'headphones', 'tablet', 'watch']
        services = ['appointment', 'reservation', 'meeting', 'consultation']
        dates = ['tomorrow', 'next week', 'Friday', 'weekend']
        
        tasks = []
        for i in range(num_tasks):
            template = random.choice(task_templates)
            
            intent = template['intent'].format(
                site=random.choice(template['sites']),
                item=random.choice(items),
                service=random.choice(services),
                date=random.choice(dates)
            )
            
            task = {
                'task_id': f'{self.split}_task_{i:06d}',
                'intent': intent,
                'sites': template['sites'],
                'difficulty': template['difficulty'],
                'expected_steps': template['expected_steps'],
                'observation': self._generate_observation(template['difficulty'])
            }
            
            tasks.append(task)
        
        # Save generated tasks
        task_file = self.data_path / f'{self.split}_synthetic_tasks.json'
        task_file.parent.mkdir(parents=True, exist_ok=True)
        with open(task_file, 'w') as f:
            json.dump(tasks, f, indent=2)
        
        return tasks
    
    def _generate_observation(self, difficulty: str) -> str:
        """Generate synthetic observation based on difficulty"""
        if difficulty == 'easy':
            return """[1] <div text="Welcome to the website">
[2] <button text="Search" clickable>
[3] <input type="text" placeholder="Search term">"""
        elif difficulty == 'medium':
            return """[1] <div text="User Dashboard">
[2] <button text="Profile" clickable>
[3] <button text="Settings" clickable>
[4] <input type="text" placeholder="Username">
[5] <input type="password" placeholder="Password">"""
        else:
            return """[1] <div text="Complex Layout">
[2] <form>
[3] <input type="text" placeholder="Name">
[4] <input type="email" placeholder="Email">
[5] <select>
[6] <option text="Option 1">
[7] <button text="Submit" clickable>
[8] <a href="/help" text="Need help?" clickable>"""
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        return self.tasks[idx]

class WebArenaDataLoader:
    """Data loader for WebArena tasks"""
    
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 4
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        
        # Create datasets
        self.train_dataset = WebArenaDataset(data_path, split='train')
        self.eval_dataset = WebArenaDataset(data_path, split='eval')
        
        # Create samplers for distributed training
        if world_size > 1:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle
            )
            shuffle = False  # Sampler handles shuffling
        else:
            self.train_sampler = None
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size // world_size,
            shuffle=shuffle and self.train_sampler is None,
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        self.sampler = self.train_sampler
    
    def _collate_fn(self, batch):
        """Custom collate function to handle task dictionaries"""
        return batch  # Return list of tasks as-is
    
    def __iter__(self):
        return iter(self.train_loader)
    
    def __len__(self):
        return len(self.train_loader)
    
    def get_eval_tasks(self, num_tasks: int = 50) -> List[Dict]:
        """Get evaluation tasks"""
        tasks = []
        for i, task in enumerate(self.eval_dataset):
            if i >= num_tasks:
                break
            tasks.append(task)
        return tasks
```

### 3.3 Create checkpoint_manager.py

```python
# training/checkpoint_manager.py
import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        rank: int = 0,
        keep_last_n: int = 5
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.rank = rank
        self.keep_last_n = keep_last_n
        
        # Create checkpoint directory
        if self.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        mas,
        episode: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint"""
        if self.rank != 0:
            return  # Only save on main process
        
        checkpoint = {
            'episode': episode,
            'method': mas.method,
            'metrics': metrics,
            'mas_config': {
                'method': mas.method,
                'budget': mas.budget,
                'num_agents': mas.num_agents,
                'use_llm_orchestrator': mas.use_llm_orchestrator
            }
        }
        
        # Save model states
        if hasattr(mas.algorithm, 'state_dict'):
            checkpoint['algorithm_state'] = mas.algorithm.state_dict()
        elif hasattr(mas.algorithm, 'save_checkpoint'):
            # Custom save method for some algorithms
            algorithm_path = self.checkpoint_dir / f'algorithm_ep{episode}.pt'
            mas.algorithm.save_checkpoint(str(algorithm_path))
            checkpoint['algorithm_path'] = str(algorithm_path.relative_to(self.checkpoint_dir))
        
        # Save orchestrator state if using neural orchestrator
        if hasattr(mas, 'orchestrator') and mas.orchestrator is not None:
            checkpoint['orchestrator_state'] = mas.orchestrator.state_dict()
        
        # Save metrics history
        checkpoint['metrics_history'] = mas.metrics
        
        # Determine checkpoint path
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_checkpoint.pt'
            # Also save a copy with episode number
            episode_path = self.checkpoint_dir / f'best_ep{episode}.pt'
            torch.save(checkpoint, episode_path)
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_ep{episode}.pt'
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'episode': episode,
                'metrics': metrics,
                'is_best': is_best
            }, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Clean up old checkpoints
        if not is_best:
            self._cleanup_old_checkpoints()
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load checkpoint"""
        if checkpoint_path is None:
            # Load best checkpoint by default
            checkpoint_path = self.checkpoint_dir / 'best_checkpoint.pt'
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the last N"""
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_ep*.pt'),
            key=lambda p: p.stat().st_mtime
        )
        
        if len(checkpoints) > self.keep_last_n:
            for checkpoint in checkpoints[:-self.keep_last_n]:
                checkpoint.unlink()
                # Also remove metadata
                metadata = checkpoint.with_suffix('.json')
                if metadata.exists():
                    metadata.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
```

---

## 🌐 Section 4: API Server

### 4.1 Create server.py

```python
# api/server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import torch
import asyncio
import uvicorn
import logging
import time
import os
import sys
from pathlib import Path

# Add workspace to path
sys.path.append('/workspace')
sys.path.append('/workspace/mas_webarena')

from mas.enhanced_webarena_mas import EnhancedWebArenaMAS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="WebArena MAS API",
    description="API for WebArena Multi-Agent System with LLM Orchestrator",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
mas_model = None
model_lock = asyncio.Lock()

# Request/Response models
class TaskRequest(BaseModel):
    intent: str = Field(..., description="Task intent/goal")
    sites: List[str] = Field(default=["example.com"], description="Target websites")
    budget: float = Field(default=1.0, gt=0, description="Budget constraint")
    method: str = Field(default="p3o", description="RL method to use")
    use_replanning: bool = Field(default=True, description="Enable dynamic replanning")
    expected_steps: int = Field(default=5, description="Expected number of steps")

class TaskResponse(BaseModel):
    success: bool
    cost: float
    reward: float
    dag: Dict[str, Any]
    trajectory: List[Dict[str, Any]]
    replanning_count: int
    execution_time: float
    method_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    available_methods: List[str]
    gpu_available: bool
    gpu_count: int

# Dependency for getting model
async def get_model():
    if mas_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return mas_model

@app.on_event("startup")
async def startup():
    """Load model on startup"""
    global mas_model
    
    try:
        checkpoint_path = os.environ.get(
            'MODEL_PATH',
            '/workspace/models/best_checkpoint.pt'
        )
        
        if Path(checkpoint_path).exists():
            logger.info(f"Loading model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
            
            # Create MAS instance
            mas_model = EnhancedWebArenaMAS(
                method=checkpoint.get('method', 'p3o'),
                budget=1.0,
                use_llm_orchestrator=True,
                device='cuda:0' if torch.cuda.is_available() else 'cpu'
            )
            
            # Load model weights if available
            if 'algorithm_state' in checkpoint:
                mas_model.algorithm.load_state_dict(checkpoint['algorithm_state'])
            
            logger.info("Model loaded successfully")
        else:
            logger.warning("No checkpoint found, creating new model")
            mas_model = EnhancedWebArenaMAS(
                method='p3o',
                budget=1.0,
                use_llm_orchestrator=True,
                device='cuda:0' if torch.cuda.is_available() else 'cpu'
            )
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        mas_model = None

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global mas_model
    mas_model = None

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "WebArena MAS API", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if mas_model is not None else "unhealthy",
        model_loaded=mas_model is not None,
        available_methods=["p3o", "ppo_lagrangian", "macpo"],
        gpu_available=torch.cuda.is_available(),
        gpu_count=torch.cuda.device_count()
    )

@app.post("/solve_task", response_model=TaskResponse)
async def solve_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    model=Depends(get_model)
):
    """Solve a WebArena task"""
    
    start_time = time.time()
    
    # Prepare task
    task = {
        'intent': request.intent,
        'sites': request.sites,
        'expected_steps': request.expected_steps,
        'difficulty': 'medium'  # Could be inferred from intent
    }
    
    # Use model lock to prevent concurrent modifications
    async with model_lock:
        # Update model configuration if needed
        if request.method != model.method:
            logger.warning(f"Method mismatch: requested {request.method}, model uses {model.method}")
        
        # Update budget
        model.budget_tracker.reset(request.budget)
        
        # Execute task (run in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, model.solve_task, task)
    
    execution_time = time.time() - start_time
    
    # Log execution in background
    background_tasks.add_task(
        log_task_execution,
        request,
        result,
        execution_time
    )
    
    return TaskResponse(
        success=result['success'],
        cost=result['cost'],
        reward=result['reward'],
        dag=result.get('dag', {}),
        trajectory=result.get('trajectory', []),
        replanning_count=result.get('replanning_count', 0),
        execution_time=execution_time,
        method_info=result.get('method_info', {})
    )

@app.post("/batch_solve", response_model=List[TaskResponse])
async def batch_solve(
    requests: List[TaskRequest],
    model=Depends(get_model)
):
    """Solve multiple tasks in batch"""
    
    responses = []
    
    for request in requests:
        response = await solve_task(request, BackgroundTasks(), model)
        responses.append(response)
    
    return responses

def log_task_execution(
    request: TaskRequest,
    result: Dict[str, Any],
    execution_time: float
):
    """Log task execution for monitoring"""
    log_entry = {
        'timestamp': time.time(),
        'request': request.dict(),
        'result': {
            'success': result['success'],
            'cost': result['cost'],
            'reward': result['reward'],
            'replanning_count': result.get('replanning_count', 0)
        },
        'execution_time': execution_time
    }
    
    # Log to file or database
    logger.info(f"Task execution: {log_entry}")

def main():
    """Run the API server"""
    port = int(os.environ.get('API_PORT', 8000))
    workers = int(os.environ.get('API_WORKERS', 4))
    
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
```

---

## 📋 Section 5: Configuration Files

### 5.1 Create .env.example

```bash
# .env.example
# API Keys (Required for LLM orchestrator)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
WANDB_API_KEY=your_wandb_api_key_here
RUNPOD_API_KEY=your_runpod_api_key_here

# Docker Registry (optional)
DOCKER_REGISTRY=your-dockerhub-username

# Training Configuration
METHOD=p3o
BUDGET=1.0
NUM_AGENTS=4
LLM_MODEL=gpt-4-turbo
ENABLE_REPLANNING=true

# Infrastructure
CUDA_VISIBLE_DEVICES=0,1,2,3
MASTER_ADDR=localhost
MASTER_PORT=12355

# Monitoring
WANDB_PROJECT=webarena-mas
TENSORBOARD_PORT=6006

# Redis (for distributed training)
REDIS_URL=redis://localhost:6379

# API Server
API_PORT=8000
API_WORKERS=4
MODEL_PATH=/workspace/models/best_checkpoint.pt
```

### 5.2 Create requirements-docker.txt

```txt
# requirements-docker.txt
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.2.0

# Deep Learning
transformers>=4.35.0
accelerate>=0.24.0
datasets>=2.14.0
tokenizers>=0.14.0

# LLM APIs
openai>=1.3.0
anthropic>=0.7.0
google-generativeai>=0.3.0

# RL and MAS
gymnasium>=0.29.0
stable-baselines3>=2.1.0
ray[tune]>=2.8.0

# Web automation
selenium>=4.15.0
playwright>=1.40.0
beautifulsoup4>=4.12.0
requests>=2.31.0

# API and serving
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
python-multipart>=0.0.6

# Monitoring and logging
wandb>=0.16.0
tensorboard>=2.15.0
prometheus-client>=0.19.0
grafana-api>=1.0.3

# Infrastructure
redis>=5.0.0
celery>=5.3.0
docker>=7.0.0
kubernetes>=28.1.0

# Utilities
python-dotenv>=1.0.0
click>=8.1.0
rich>=13.7.0
tqdm>=4.66.0
pyyaml>=6.0.1
networkx>=3.2.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# RunPod SDK
runpod>=1.4.0
```

### 5.3 Create training config files

```yaml
# config/test_training.yaml
# Quick test configuration
method: p3o
budget: 1.0
num_agents: 2
llm_model: gpt-4-turbo
use_llm_orchestrator: true
enable_replanning: true

# Training parameters
episodes: 100
batch_size: 8
learning_rate: 3e-4
gamma: 0.99
ppo_epochs: 4

# Infrastructure
gpu_type: "NVIDIA RTX 3090"
gpu_count: 1
distributed: false

# Data
data_path: /workspace/data
eval_interval: 20
save_interval: 50

# Logging
use_wandb: true
wandb_project: webarena-mas-test
run_name: test-run
```

```yaml
# config/standard_training.yaml
# Standard training configuration
method: p3o
budget: 1.0
num_agents: 4
llm_model: gpt-4-turbo
use_llm_orchestrator: true
enable_replanning: true

# Training parameters
episodes: 10000
batch_size: 32
learning_rate: 3e-4
gamma: 0.99
ppo_epochs: 4

# Infrastructure
gpu_type: "NVIDIA RTX A6000"
gpu_count: 2
distributed: true

# Data
data_path: /workspace/data
eval_interval: 100
save_interval: 500

# Logging
use_wandb: true
wandb_project: webarena-mas
run_name: standard-training
stream_logs: true
```

```yaml
# config/full_training.yaml
# Full research training configuration
method: p3o
budget: 1.0
num_agents: 4
max_nodes: 10
llm_model: gpt-4-turbo
use_llm_orchestrator: true
enable_replanning: true

# Training parameters
episodes: 50000
batch_size: 128
learning_rate: 3e-4
gamma: 0.99
ppo_epochs: 4

# Method-specific parameters
penalty_coef: 10.0  # For P3O
alpha: 1.05
beta: 0.95

# Infrastructure
gpu_type: "NVIDIA RTX A6000"
gpu_count: 4
distributed: true
memory_gb: 64

# Data
data_path: /workspace/persistent/data
eval_interval: 100
save_interval: 1000
num_workers: 8

# Checkpointing
checkpoint_dir: /workspace/persistent/checkpoints
keep_checkpoints: 10

# Logging
use_wandb: true
wandb_project: webarena-mas-research
run_name: full-training
log_dir: /workspace/persistent/logs
stream_logs: true
```

### 5.4 Create Makefile

```makefile
# Makefile
.PHONY: help build push run train serve test clean deploy

# Variables
DOCKER_IMAGE ?= webarena-mas
VERSION ?= latest
REGISTRY ?= local
METHOD ?= p3o

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker image
	docker build -t $(REGISTRY)/$(DOCKER_IMAGE):$(VERSION) -f docker/Dockerfile .

push: ## Push Docker image to registry
	docker push $(REGISTRY)/$(DOCKER_IMAGE):$(VERSION)

run: ## Run training locally
	docker-compose -f docker/docker-compose.yml up mas-trainer

train-test: ## Run quick training test
	docker run --rm --gpus all \
		--env-file .env \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/models:/workspace/models \
		$(REGISTRY)/$(DOCKER_IMAGE):$(VERSION) \
		train --config /workspace/config/test_training.yaml

serve: ## Start API server
	docker-compose -f docker/docker-compose.yml up mas-api

test: ## Run tests
	docker run --rm \
		--env-file .env \
		$(REGISTRY)/$(DOCKER_IMAGE):$(VERSION) \
		test

deploy-runpod: ## Deploy to RunPod
	python scripts/deploy_runpod.py create \
		--config config/standard_training.yaml \
		--api-key $(RUNPOD_API_KEY)

monitor: ## Monitor training with TensorBoard
	docker run --rm -p 6006:6006 \
		-v $(PWD)/logs:/workspace/logs \
		$(REGISTRY)/$(DOCKER_IMAGE):$(VERSION) \
		tensorboard

jupyter: ## Start Jupyter notebook
	docker run --rm -p 8888:8888 \
		-v $(PWD):/workspace \
		$(REGISTRY)/$(DOCKER_IMAGE):$(VERSION) \
		jupyter

clean: ## Clean up containers and volumes
	docker-compose -f docker/docker-compose.yml down -v
	rm -rf logs/* checkpoints/* __pycache__ .pytest_cache

logs: ## View training logs
	docker-compose -f docker/docker-compose.yml logs -f mas-trainer

shell: ## Open shell in container
	docker run --rm -it --gpus all \
		--env-file .env \
		-v $(PWD):/workspace \
		$(REGISTRY)/$(DOCKER_IMAGE):$(VERSION) \
		bash
```

### 5.5 Create QUICKSTART.md

```markdown
# QUICKSTART.md - WebArena MAS Quick Start Guide

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA 11.8+ support
- At least 32GB RAM
- 100GB free disk space
- API keys for LLM providers

## Quick Setup (15 minutes)

### 1. Clone and Configure

```bash
git clone <repository>
cd mas_webarena
cp .env.example .env
```

Edit `.env` with your API keys:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
RUNPOD_API_KEY=...
WANDB_API_KEY=...
```

### 2. Build Docker Image

```bash
make build
# Or manually:
docker build -t webarena-mas -f docker/Dockerfile .
```

### 3. Quick Test

```bash
# Test that everything works
make train-test
```

### 4. Start Training

#### Local Training (1 GPU)
```bash
docker-compose up mas-trainer
```

#### Multi-GPU Training
```bash
docker run --gpus all --env-file .env \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  webarena-mas train \
    --method p3o \
    --num-gpus 4 \
    --distributed \
    --episodes 10000
```

### 5. Deploy to RunPod

```bash
# Push to Docker Hub first
docker tag webarena-mas:latest yourusername/webarena-mas:latest
docker push yourusername/webarena-mas:latest

# Deploy
python scripts/deploy_runpod.py create \
  --config config/standard_training.yaml
```

## Training Configurations

### Test Run (2 hours, ~$10)
```bash
make train-test
```

### Standard Training (24 hours, ~$200)
```bash
docker-compose up mas-trainer
```

### Full Research (3-5 days, ~$1000)
```bash
python scripts/deploy_runpod.py create \
  --config config/full_training.yaml
```

## Monitoring

### TensorBoard
```bash
make monitor
# Visit http://localhost:6006
```

### Weights & Biases
Visit https://wandb.ai/your-username/webarena-mas

### Logs
```bash
make logs
```

## API Server

### Start Server
```bash
make serve
```

### Test API
```bash
curl -X POST http://localhost:8000/solve_task \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "Search for laptops on shopping.com",
    "sites": ["shopping.com"],
    "budget": 1.0
  }'
```

## Common Issues

### GPU Not Found
```bash
# Check CUDA installation
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
```

### Out of Memory
Reduce batch size in config:
```yaml
batch_size: 16  # Instead of 32
```

### API Key Errors
Verify keys in `.env` file and test:
```python
import openai
openai.api_key = "your-key"
openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": "test"}])
```

## Cost Estimates

| Config | GPU Hours | LLM Calls | Total Cost |
|--------|-----------|-----------|------------|
| Test | 2 | ~500 | $10-20 |
| Standard | 24 | ~5K | $150-250 |
| Full | 96 | ~25K | $800-1200 |

## Support

- Documentation: See `mas_webarena/README.md`
- Issues: Create GitHub issue
- Logs: Check `/workspace/logs`
```

---

## 🚀 Implementation Instructions

### Step 1: Create Directory Structure
```bash
cd mas_webarena
mkdir -p docker scripts training api config deployment
```

### Step 2: Create All Files
Create each file in the appropriate directory as specified above.

### Step 3: Build and Test
```bash
# Build Docker image
docker build -t webarena-mas -f docker/Dockerfile .

# Test locally
docker run --rm --env-file .env webarena-mas test
```

### Step 4: Deploy
```bash
# Local deployment
docker-compose up

# RunPod deployment
python scripts/deploy_runpod.py create --config config/standard_training.yaml
```

---

## ✅ Implementation Checklist

- [ ] Create all directories
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml
- [ ] Create entrypoint.sh
- [ ] Create train_distributed.py
- [ ] Create deploy_runpod.py
- [ ] Create distributed_trainer.py
- [ ] Create data_loader.py
- [ ] Create checkpoint_manager.py
- [ ] Create API server.py
- [ ] Create configuration files
- [ ] Create .env.example
- [ ] Create Makefile
- [ ] Create QUICKSTART.md
- [ ] Build Docker image
- [ ] Test locally
- [ ] Push to registry
- [ ] Deploy to RunPod

---

This complete implementation provides everything needed to deploy and train your WebArena MAS system with Docker and RunPod. After Claude implements these files, you'll be able to immediately start training with your LLM APIs.