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


class MACPO:
    """Multi-Agent Constrained Policy Optimization baseline"""
    
    def __init__(self, 
                 num_agents: int = 4,
                 state_dim: int = 128,
                 action_dim: int = 32,
                 lr_policy: float = 3e-4,
                 lr_critic: float = 3e-4,
                 lr_lambda: float = 1e-3,
                 budget: float = 1.0,
                 alpha: float = 1.05,
                 clip_ratio: float = 0.2,
                 ppo_epochs: int = 4,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 coordination_weight: float = 0.1):
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Per-agent networks
        self.policies = nn.ModuleList([
            PolicyNetwork(state_dim, action_dim) for _ in range(num_agents)
        ])
        self.critics = nn.ModuleList([
            DualCriticNetwork(state_dim) for _ in range(num_agents)
        ])
        
        # Initialize weights
        for policy in self.policies:
            policy.apply(initialize_weights)
        for critic in self.critics:
            critic.apply(initialize_weights)
        
        # Per-agent optimizers
        self.policy_optimizers = [
            optim.Adam(policy.parameters(), lr=lr_policy) 
            for policy in self.policies
        ]
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=lr_critic) 
            for critic in self.critics
        ]
        
        # Shared Lagrange multiplier for global constraint
        self.lambda_coord = torch.tensor(0.1, requires_grad=True)
        self.optimizer_lambda = optim.Adam([self.lambda_coord], lr=lr_lambda)
        
        # Hyperparameters
        self.budget = budget
        self.alpha = alpha
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.coordination_weight = coordination_weight
        
        # Tracking
        self.agent_costs = [[] for _ in range(num_agents)]
        self.agent_rewards = [[] for _ in range(num_agents)]
        self.coordination_history = []
        self.lambda_history = []
        
    def update(self, trajectories_by_agent: List[List[Dict]]) -> Dict[str, float]:
        """MACPO update with coordination constraints"""
        
        if len(trajectories_by_agent) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} agent trajectories, got {len(trajectories_by_agent)}")
        
        agent_updates = {}
        total_cost = 0
        total_reward = 0
        coordination_loss = 0
        
        # Convert trajectories to tensors for each agent
        agent_data = []
        for agent_id, trajectories in enumerate(trajectories_by_agent):
            states = torch.FloatTensor([t['state'] for t in trajectories])
            actions = torch.LongTensor([t['action'] for t in trajectories])
            rewards = torch.FloatTensor([t['reward'] for t in trajectories])
            costs = torch.FloatTensor([t['cost'] for t in trajectories])
            dones = torch.BoolTensor([t.get('done', False) for t in trajectories])
            
            agent_data.append({
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'costs': costs,
                'dones': dones
            })
            
            total_cost += costs.mean().item()
            total_reward += rewards.mean().item()
        
        # Average cost across all agents
        avg_total_cost = total_cost / self.num_agents
        
        # Update each agent
        for agent_id in range(self.num_agents):
            data = agent_data[agent_id]
            
            # Compute values and advantages for this agent
            with torch.no_grad():
                reward_values, cost_values = self.critics[agent_id](data['states'])
            
            reward_advantages = self._compute_gae(data['rewards'], reward_values, data['dones'])
            cost_advantages = self._compute_gae(data['costs'], cost_values, data['dones'])
            
            reward_returns = reward_advantages + reward_values
            cost_returns = cost_advantages + cost_values
            
            # Get old policy probabilities
            with torch.no_grad():
                old_probs, old_log_probs = self.policies[agent_id].get_action_probs(data['states'])
                old_action_log_probs = old_log_probs.gather(1, data['actions'].unsqueeze(-1)).squeeze(-1)
            
            # PPO updates for this agent
            policy_losses = []
            critic_losses = []
            
            for epoch in range(self.ppo_epochs):
                # Policy update with coordination
                probs, log_probs = self.policies[agent_id].get_action_probs(data['states'])
                action_log_probs = log_probs.gather(1, data['actions'].unsqueeze(-1)).squeeze(-1)
                
                # Compute ratio
                ratio = torch.exp(action_log_probs - old_action_log_probs)
                
                # Individual reward objective
                surr1 = ratio * reward_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * reward_advantages
                reward_objective = torch.min(surr1, surr2).mean()
                
                # Global constraint term (shared lambda for coordination)
                # Each agent considers the global cost constraint
                global_cost_term = self.lambda_coord.detach() * cost_advantages.mean()
                
                # Coordination penalty: encourage similar action distributions across agents
                if self.coordination_weight > 0:
                    coord_penalty = self._compute_coordination_penalty(agent_id, data['states'], agent_data)
                else:
                    coord_penalty = 0
                
                # Combined policy loss
                policy_loss = -(reward_objective - global_cost_term - self.coordination_weight * coord_penalty)
                
                # Update policy
                self.policy_optimizers[agent_id].zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policies[agent_id].parameters(), 0.5)
                self.policy_optimizers[agent_id].step()
                policy_losses.append(policy_loss.item())
                
                # Critic update
                reward_pred, cost_pred = self.critics[agent_id](data['states'])
                
                reward_loss = F.mse_loss(reward_pred, reward_returns)
                cost_loss = F.mse_loss(cost_pred, cost_returns)
                critic_loss = reward_loss + cost_loss
                
                self.critic_optimizers[agent_id].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 0.5)
                self.critic_optimizers[agent_id].step()
                critic_losses.append(critic_loss.item())
            
            # Track per-agent metrics
            agent_updates[f'agent_{agent_id}_policy_loss'] = np.mean(policy_losses)
            agent_updates[f'agent_{agent_id}_critic_loss'] = np.mean(critic_losses)
            agent_updates[f'agent_{agent_id}_reward'] = data['rewards'].mean().item()
            agent_updates[f'agent_{agent_id}_cost'] = data['costs'].mean().item()
            
            self.agent_rewards[agent_id].append(data['rewards'].mean().item())
            self.agent_costs[agent_id].append(data['costs'].mean().item())
        
        # Update shared Lagrange multiplier
        constraint_violation = avg_total_cost - self.alpha * self.budget
        
        self.optimizer_lambda.zero_grad()
        lambda_loss = -self.lambda_coord * constraint_violation
        lambda_loss.backward()
        self.optimizer_lambda.step()
        
        # Project lambda to be non-negative
        with torch.no_grad():
            self.lambda_coord.clamp_(min=0, max=100.0)
        
        self.lambda_history.append(self.lambda_coord.item())
        
        # Compute coordination metrics
        coordination_metric = self._compute_coordination_metric(agent_data)
        self.coordination_history.append(coordination_metric)
        
        return {
            'reward': total_reward / self.num_agents,
            'cost': avg_total_cost,
            'lambda_coord': self.lambda_coord.item(),
            'constraint_violation': constraint_violation.item(),
            'cost_guarantee_satisfied': avg_total_cost <= self.alpha * self.budget,
            'coordination_metric': coordination_metric,
            **agent_updates
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
    
    def _compute_coordination_penalty(self, agent_id: int, states: torch.Tensor, 
                                    all_agent_data: List[Dict]) -> torch.Tensor:
        """Compute coordination penalty to encourage similar behaviors"""
        if self.num_agents == 1:
            return torch.tensor(0.0)
        
        # Get current agent's action probabilities
        current_probs = self.policies[agent_id](states)
        
        # Compare with other agents' action probabilities on same states
        kl_divergences = []
        for other_id in range(self.num_agents):
            if other_id != agent_id:
                with torch.no_grad():
                    other_probs = self.policies[other_id](states)
                
                # KL divergence between action distributions
                kl_div = F.kl_div(
                    F.log_softmax(current_probs, dim=-1),
                    F.softmax(other_probs, dim=-1),
                    reduction='batchmean'
                )
                kl_divergences.append(kl_div)
        
        if kl_divergences:
            return torch.mean(torch.stack(kl_divergences))
        else:
            return torch.tensor(0.0)
    
    def _compute_coordination_metric(self, agent_data: List[Dict]) -> float:
        """Compute coordination metric (lower is better coordinated)"""
        if self.num_agents == 1:
            return 0.0
        
        # Compute pairwise policy similarity
        similarities = []
        
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                # Sample states from both agents
                states_i = agent_data[i]['states'][:min(50, len(agent_data[i]['states']))]
                
                with torch.no_grad():
                    probs_i = self.policies[i](states_i)
                    probs_j = self.policies[j](states_i)
                
                # Jensen-Shannon divergence (symmetric)
                m = 0.5 * (probs_i + probs_j)
                js_div = 0.5 * F.kl_div(F.log_softmax(probs_i, dim=-1), m, reduction='batchmean') + \
                         0.5 * F.kl_div(F.log_softmax(probs_j, dim=-1), m, reduction='batchmean')
                
                similarities.append(js_div.item())
        
        return np.mean(similarities) if similarities else 0.0
    
    def get_action(self, agent_id: int, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from specific agent's policy"""
        with torch.no_grad():
            if deterministic:
                probs = self.policies[agent_id](state)
                action = torch.argmax(probs, dim=-1)
                log_prob = torch.log(probs.max(dim=-1)[0] + 1e-8)
            else:
                action, log_prob = self.policies[agent_id].sample_action(state)
            
            return action, log_prob
    
    def get_value(self, agent_id: int, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get reward and cost values from specific agent's critic"""
        with torch.no_grad():
            return self.critics[agent_id](state)
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'policies': [policy.state_dict() for policy in self.policies],
            'critics': [critic.state_dict() for critic in self.critics],
            'lambda_coord': self.lambda_coord,
            'coordination_history': self.coordination_history,
            'lambda_history': self.lambda_history
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        
        for i, policy_state in enumerate(checkpoint['policies']):
            self.policies[i].load_state_dict(policy_state)
        
        for i, critic_state in enumerate(checkpoint['critics']):
            self.critics[i].load_state_dict(critic_state)
        
        self.lambda_coord = checkpoint['lambda_coord']
        self.coordination_history = checkpoint['coordination_history']
        self.lambda_history = checkpoint['lambda_history']