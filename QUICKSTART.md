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