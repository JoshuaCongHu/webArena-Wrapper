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