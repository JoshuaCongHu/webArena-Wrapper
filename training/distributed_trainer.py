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