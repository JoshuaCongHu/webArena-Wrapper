import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.networks import PolicyNetwork, DualCriticNetwork, initialize_weights


class PPOLagrangian:
    """PPO with Lagrangian constraint handling (dual method)"""
    
    def __init__(self, 
                 state_dim: int = 128,
                 action_dim: int = 64,
                 lr_policy: float = 3e-4,
                 lr_critic: float = 3e-4,
                 lr_lambda: float = 1e-3,
                 budget: float = 1.0,
                 alpha: float = 1.05,  # Budget margin
                 beta: float = 0.95,   # Guarantee rate
                 clip_ratio: float = 0.2,
                 ppo_epochs: int = 4,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 max_lambda: float = 100.0):
        
        # Networks
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.critic_net = DualCriticNetwork(state_dim)
        
        # Initialize weights
        self.policy_net.apply(initialize_weights)
        self.critic_net.apply(initialize_weights)
        
        # Optimizers
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr=lr_critic)
        
        # Lagrange multiplier - initialized to small positive value
        self.lambda_param = torch.tensor(0.1, requires_grad=True)
        self.optimizer_lambda = optim.Adam([self.lambda_param], lr=lr_lambda)
        
        # Hyperparameters
        self.budget = budget
        self.alpha = alpha
        self.beta = beta
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.max_lambda = max_lambda
        
        # Tracking
        self.duality_gap_history = []
        self.lambda_history = []
        self.constraint_violation_history = []
        
    def update(self, trajectories: List[Dict]) -> Dict[str, float]:
        """Update policy using PPO-Lagrangian"""
        
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
        
        # Get old policy probabilities
        with torch.no_grad():
            old_probs, old_log_probs = self.policy_net.get_action_probs(states)
            old_action_log_probs = old_log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # PPO updates
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
            
            # Lagrangian objective (reward - lambda * cost)
            lagrangian_advantages = reward_advantages - self.lambda_param.detach() * cost_advantages
            
            # Policy loss with constraint
            policy_loss = -torch.min(
                ratio * lagrangian_advantages,
                torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * lagrangian_advantages
            ).mean()
            
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
        
        # Update Lagrange multiplier
        constraint_violation = costs.mean() - self.alpha * self.budget
        
        # Lambda gradient ascent (dual ascent)
        self.optimizer_lambda.zero_grad()
        lambda_loss = -self.lambda_param * constraint_violation.detach()
        lambda_loss.backward()
        self.optimizer_lambda.step()
        
        # Project lambda to [0, max_lambda]
        with torch.no_grad():
            self.lambda_param.clamp_(min=0, max=self.max_lambda)
        
        # Calculate duality gap
        primal_value = rewards.mean().item()
        dual_value = rewards.mean().item() - self.lambda_param.item() * (costs.mean().item() - self.alpha * self.budget)
        duality_gap = abs(primal_value - dual_value)
        
        # Track metrics
        self.duality_gap_history.append(duality_gap)
        self.lambda_history.append(self.lambda_param.item())
        self.constraint_violation_history.append(constraint_violation.item())
        
        return {
            'reward': rewards.mean().item(),
            'cost': costs.mean().item(),
            'lambda': self.lambda_param.item(),
            'duality_gap': duality_gap,
            'constraint_violation': constraint_violation.item(),
            'policy_loss': np.mean(policy_losses),
            'critic_loss': np.mean(critic_losses),
            'cost_guarantee_satisfied': costs.mean().item() <= self.alpha * self.budget
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
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'critic_net': self.critic_net.state_dict(),
            'lambda_param': self.lambda_param,
            'duality_gap_history': self.duality_gap_history,
            'lambda_history': self.lambda_history
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.critic_net.load_state_dict(checkpoint['critic_net'])
        self.lambda_param = checkpoint['lambda_param']
        self.duality_gap_history = checkpoint['duality_gap_history']
        self.lambda_history = checkpoint['lambda_history']