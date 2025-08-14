import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class PolicyNetwork(nn.Module):
    """Base policy network for all algorithms"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action logits"""
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        action_logits = self.action_head(x)
        return F.softmax(action_logits, dim=-1)
    
    def get_action_probs(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action probabilities and log probabilities"""
        probs = self.forward(state)
        log_probs = torch.log(probs + 1e-8)  # Add epsilon for numerical stability
        return probs, log_probs
    
    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        probs, log_probs = self.get_action_probs(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)


class ValueNetwork(nn.Module):
    """Value function network"""
    
    def __init__(self, 
                 state_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state value"""
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        value = self.value_head(x)
        return value.squeeze(-1)


class DualCriticNetwork(nn.Module):
    """Dual critic for cost and reward estimation"""
    
    def __init__(self, 
                 state_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        # Shared layers
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Reward critic
        self.reward_fc = nn.Linear(hidden_dim, hidden_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        
        # Cost critic
        self.cost_fc = nn.Linear(hidden_dim, hidden_dim)
        self.cost_head = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both reward and cost values"""
        # Shared features
        x = F.relu(self.shared_fc1(state))
        x = self.dropout(x)
        x = F.relu(self.shared_fc2(x))
        x = self.dropout(x)
        
        # Reward branch
        reward_x = F.relu(self.reward_fc(x))
        reward_value = self.reward_head(reward_x).squeeze(-1)
        
        # Cost branch
        cost_x = F.relu(self.cost_fc(x))
        cost_value = self.cost_head(cost_x).squeeze(-1)
        
        return reward_value, cost_value


def initialize_weights(module):
    """Initialize network weights"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        nn.init.constant_(module.bias, 0)