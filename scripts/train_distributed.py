#!/usr/bin/env python3
# scripts/train_distributed.py
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