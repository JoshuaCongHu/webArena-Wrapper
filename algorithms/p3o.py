import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.networks import PolicyNetwork, DualCriticNetwork, initialize_weights


class P3O:
    """Penalized PPO (P3O) - Primal-only constraint handling"""
    
    def __init__(self,
                 state_dim: int = 128,
                 action_dim: int = 64,
                 lr_policy: float = 3e-4,
                 lr_critic: float = 3e-4,
                 budget: float = 1.0,
                 penalty_coef: float = 10.0,
                 alpha: float = 1.05,  # Budget margin
                 clip_ratio: float = 0.2,
                 ppo_epochs: int = 4,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 adaptive_penalty: bool = True,
                 penalty_lr: float = 0.01):
        
        # Networks
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.critic_net = DualCriticNetwork(state_dim)
        
        # Initialize weights
        self.policy_net.apply(initialize_weights)
        self.critic_net.apply(initialize_weights)
        
        # Optimizers
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr=lr_critic)
        
        # Hyperparameters
        self.budget = budget
        self.penalty_coef = penalty_coef
        self.alpha = alpha
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        
        # Adaptive penalty mechanism
        self.adaptive_penalty = adaptive_penalty
        self.penalty_lr = penalty_lr
        
        # Tracking
        self.penalty_history = []
        self.constraint_violation_history = []
        self.cost_history = []
        
    def update(self, trajectories: List[Dict]) -> Dict[str, float]:
        """Update using penalized objective (no dual variable)"""
        
        # Convert trajectories to tensors
        states = torch.FloatTensor([t['state'] for t in trajectories])
        actions = torch.LongTensor([t['action'] for t in trajectories])
        rewards = torch.FloatTensor([t['reward'] for t in trajectories])
        costs = torch.FloatTensor([t['cost'] for t in trajectories])
        dones = torch.BoolTensor([t.get('done', False) for t in trajectories])
        
        # Compute values and advantages
        with torch.no_grad():
            reward_values, cost_values = self.critic_net(states)
        
        # Compute GAE for rewards and costs
        reward_advantages = self._compute_gae(rewards, reward_values, dones)
        cost_advantages = self._compute_gae(costs, cost_values, dones)
        
        # Compute returns
        reward_returns = reward_advantages + reward_values
        cost_returns = cost_advantages + cost_values
        
        # Constraint penalty (quadratic for smooth gradients)
        avg_cost = costs.mean()
        constraint_violation = torch.relu(avg_cost - self.alpha * self.budget)
        penalty = self.penalty_coef * constraint_violation ** 2
        
        # Get old policy probabilities
        with torch.no_grad():
            old_probs, old_log_probs = self.policy_net.get_action_probs(states)
            old_action_log_probs = old_log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # PPO updates with penalty
        policy_losses = []
        critic_losses = []
        
        for epoch in range(self.ppo_epochs):
            # Policy update
            probs, log_probs = self.policy_net.get_action_probs(states)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            
            # Compute ratio
            ratio = torch.exp(action_log_probs - old_action_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * reward_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * reward_advantages
            
            # P3O objective: maximize reward, minimize cost penalty
            # Note: We subtract penalty from the objective (equivalent to adding cost penalty)
            clipped_objective = torch.min(surr1, surr2).mean()
            
            # Recompute penalty for this batch (may have changed due to policy update)
            current_avg_cost = costs.mean()
            current_violation = torch.relu(current_avg_cost - self.alpha * self.budget)
            current_penalty = self.penalty_coef * current_violation ** 2
            
            # Direct penalty in objective (no lambda)
            policy_loss = -(clipped_objective - current_penalty)
            
            # Update policy
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer_policy.step()
            policy_losses.append(policy_loss.item())
            
            # Critic update
            reward_pred, cost_pred = self.critic_net(states)
            
            reward_loss = F.mse_loss(reward_pred, reward_returns)
            cost_loss = F.mse_loss(cost_pred, cost_returns)
            critic_loss = reward_loss + cost_loss
            
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
            self.optimizer_critic.step()
            critic_losses.append(critic_loss.item())
        
        # Adaptive penalty coefficient adjustment
        if self.adaptive_penalty:
            violation_rate = (costs > self.alpha * self.budget).float().mean().item()
            if violation_rate > 0.1:  # Too many violations
                self.penalty_coef *= (1 + self.penalty_lr)
            elif violation_rate < 0.05:  # Too conservative
                self.penalty_coef *= (1 - self.penalty_lr * 0.5)
            
            # Keep penalty coefficient in reasonable range
            self.penalty_coef = np.clip(self.penalty_coef, 1.0, 1000.0)
        
        # Track metrics
        self.penalty_history.append(penalty.item())
        self.constraint_violation_history.append(constraint_violation.item())
        self.cost_history.append(avg_cost.item())
        
        return {
            'reward': rewards.mean().item(),
            'cost': avg_cost.item(),
            'penalty': penalty.item(),
            'penalty_coef': self.penalty_coef,
            'constraint_violation': constraint_violation.item(),
            'policy_loss': np.mean(policy_losses),
            'critic_loss': np.mean(critic_losses),
            'cost_guarantee_satisfied': avg_cost.item() <= self.alpha * self.budget,
            'violation_rate': (costs > self.alpha * self.budget).float().mean().item()
        }
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                     dones: torch.Tensor) -> torch.Tensor:
        """Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 0
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t + 1].float()
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lambda_gae * next_non_terminal * last_advantage
            
        return advantages
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from policy"""
        with torch.no_grad():
            if deterministic:
                probs = self.policy_net(state)
                action = torch.argmax(probs, dim=-1)
                log_prob = torch.log(probs.max(dim=-1)[0] + 1e-8)
            else:
                action, log_prob = self.policy_net.sample_action(state)
            
            return action, log_prob
    
    def get_value(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get reward and cost values"""
        with torch.no_grad():
            return self.critic_net(state)
    
    def set_penalty_coefficient(self, penalty_coef: float):
        """Manually set penalty coefficient"""
        self.penalty_coef = penalty_coef
    
    def get_penalty_stats(self) -> Dict[str, float]:
        """Get penalty-related statistics"""
        if not self.penalty_history:
            return {}
        
        return {
            'avg_penalty': np.mean(self.penalty_history[-100:]),  # Last 100 episodes
            'avg_violation': np.mean(self.constraint_violation_history[-100:]),
            'avg_cost': np.mean(self.cost_history[-100:]),
            'current_penalty_coef': self.penalty_coef
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'critic_net': self.critic_net.state_dict(),
            'penalty_coef': self.penalty_coef,
            'penalty_history': self.penalty_history,
            'constraint_violation_history': self.constraint_violation_history
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.critic_net.load_state_dict(checkpoint['critic_net'])
        self.penalty_coef = checkpoint['penalty_coef']
        self.penalty_history = checkpoint['penalty_history']
        self.constraint_violation_history = checkpoint['constraint_violation_history']