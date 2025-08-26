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