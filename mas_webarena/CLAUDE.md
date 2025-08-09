# Claude MAS WebArena - Complete Research Implementation Plan

## Executive Summary
Implementation of a **Constrained Multi-Agent Reinforcement Learning System** for WebArena with:
- **Novel Contributions**: CMDP formulation, end-to-end DAG learning, primal vs dual constraint methods comparison
- **Core Innovation**: PPO-Lagrangian vs P3O (Penalized PPO) for cost-guaranteed web automation
- **Positioning**: Hierarchical RL framework with emergent coordination strategies
- **Target**: NeurIPS/ICML main conference (not just workshop)

## 1. Theoretical Foundation

### 1.1 Constrained MDP Formulation
```python
# State space
S = (observation, history, budget_remaining, graph_state)

# Action space  
A = (dag_decomposition, agent_assignment, execution_mode)

# Constraint
C: P(cost_episode ≤ 1.05 * budget) ≥ 0.95

# Objective
maximize E[R] subject to C
```

### 1.2 Hierarchical RL Framing
- **High-level policy** (Orchestrator): Generates macro-actions (DAG + agent assignments)
- **Low-level policies** (Agents): Execute primitive web actions
- **Temporal abstraction**: DAG represents extended action sequences

## 2. Core Algorithms Implementation

### 2.1 Dual Method - PPO-Lagrangian
**File:** `algorithms/ppo_lagrangian.py`
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional

class PPOLagrangian:
    """PPO with Lagrangian constraint handling (dual method)"""
    
    def __init__(self, 
                 state_dim: int = 128,
                 action_dim: int = 64,
                 lr_policy: float = 3e-4,
                 lr_lambda: float = 1e-3,
                 budget: float = 1.0,
                 alpha: float = 1.05,  # Budget margin
                 beta: float = 0.95):   # Guarantee rate
        
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.cost_value_net = ValueNetwork(state_dim)  # For cost estimation
        
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr_policy)
        self.optimizer_cost = optim.Adam(self.cost_value_net.parameters(), lr=lr_policy)
        
        # Lagrange multiplier
        self.lambda_param = torch.tensor(0.0, requires_grad=True)
        self.optimizer_lambda = optim.Adam([self.lambda_param], lr=lr_lambda)
        
        self.budget = budget
        self.alpha = alpha
        self.beta = beta
        
        # Tracking
        self.duality_gap_history = []
        
    def update(self, trajectories: List[Dict]) -> Dict[str, float]:
        """Update policy using PPO-Lagrangian"""
        
        states = torch.FloatTensor([t['state'] for t in trajectories])
        actions = torch.LongTensor([t['action'] for t in trajectories])
        rewards = torch.FloatTensor([t['reward'] for t in trajectories])
        costs = torch.FloatTensor([t['cost'] for t in trajectories])
        
        # Compute advantages
        values = self.value_net(states)
        cost_values = self.cost_value_net(states)
        advantages = self.compute_gae(rewards, values)
        cost_advantages = self.compute_gae(costs, cost_values)
        
        # PPO update
        for _ in range(4):  # PPO epochs
            # Policy loss with constraint
            action_probs = self.policy_net(states)
            ratio = action_probs / action_probs.detach()
            
            L_clip = torch.min(
                ratio * advantages,
                torch.clamp(ratio, 0.8, 1.2) * advantages
            )
            
            # Lagrangian objective
            L_policy = L_clip.mean() - self.lambda_param * cost_advantages.mean()
            
            self.optimizer_policy.zero_grad()
            (-L_policy).backward()
            self.optimizer_policy.step()
            
        # Update Lagrange multiplier
        constraint_violation = costs.mean() - self.alpha * self.budget
        
        self.optimizer_lambda.zero_grad()
        lambda_loss = -self.lambda_param * constraint_violation
        lambda_loss.backward()
        self.optimizer_lambda.step()
        
        # Project lambda to be non-negative
        with torch.no_grad():
            self.lambda_param.clamp_(min=0)
        
        # Calculate duality gap
        primal_value = rewards.mean() - costs.mean()
        dual_value = rewards.mean() - self.lambda_param.item() * (costs.mean() - self.alpha * self.budget)
        duality_gap = abs(primal_value - dual_value)
        self.duality_gap_history.append(duality_gap)
        
        return {
            'reward': rewards.mean().item(),
            'cost': costs.mean().item(),
            'lambda': self.lambda_param.item(),
            'duality_gap': duality_gap,
            'constraint_violation': constraint_violation.item()
        }
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                    gamma: float = 0.99, lambda_: float = 0.95) -> torch.Tensor:
        """Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_advantage = delta + gamma * lambda_ * last_advantage
            
        return advantages
```

### 2.2 Primal Method - P3O (Penalized PPO)
**File:** `algorithms/p3o.py`
```python
class P3O:
    """Penalized PPO (P3O) - Primal-only constraint handling"""
    
    def __init__(self,
                 state_dim: int = 128,
                 action_dim: int = 64,
                 lr: float = 3e-4,
                 budget: float = 1.0,
                 penalty_coef: float = 10.0):
        
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + 
            list(self.value_net.parameters()), 
            lr=lr
        )
        
        self.budget = budget
        self.penalty_coef = penalty_coef
        
    def update(self, trajectories: List[Dict]) -> Dict[str, float]:
        """Update using penalized objective (no dual variable)"""
        
        states = torch.FloatTensor([t['state'] for t in trajectories])
        actions = torch.LongTensor([t['action'] for t in trajectories])
        rewards = torch.FloatTensor([t['reward'] for t in trajectories])
        costs = torch.FloatTensor([t['cost'] for t in trajectories])
        
        # Compute advantages
        values = self.value_net(states)
        advantages = self.compute_gae(rewards, values)
        
        # Constraint penalty (quadratic)
        constraint_violation = torch.relu(costs.mean() - 1.05 * self.budget)
        penalty = self.penalty_coef * constraint_violation ** 2
        
        # PPO update with penalty
        for _ in range(4):
            action_probs = self.policy_net(states)
            ratio = action_probs / action_probs.detach()
            
            L_clip = torch.min(
                ratio * advantages,
                torch.clamp(ratio, 0.8, 1.2) * advantages
            )
            
            # Direct penalty in objective (no lambda)
            L_p3o = L_clip.mean() - penalty
            
            self.optimizer.zero_grad()
            (-L_p3o).backward()
            self.optimizer.step()
        
        return {
            'reward': rewards.mean().item(),
            'cost': costs.mean().item(),
            'penalty': penalty.item(),
            'constraint_violation': constraint_violation.item()
        }
```

### 2.3 MACPO Baseline
**File:** `algorithms/macpo.py`
```python
class MACPO:
    """Multi-Agent Constrained Policy Optimization baseline"""
    
    def __init__(self, num_agents: int = 4, state_dim: int = 128):
        self.num_agents = num_agents
        self.policies = [PolicyNetwork(state_dim, 32) for _ in range(num_agents)]
        self.critics = [ValueNetwork(state_dim) for _ in range(num_agents)]
        self.lambda_coord = torch.tensor(0.0, requires_grad=True)
        
    def update(self, trajectories: List[Dict]) -> Dict[str, float]:
        """MACPO update with coordination constraints"""
        # Implementation of MACPO algorithm
        # Key: coordinate multiple agents with shared constraint
        pass
```

## 3. Orchestrator Architecture

### 3.1 DAG Generation Network
**File:** `models/orchestrator.py`
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class OrchestratorPolicy(nn.Module):
    """Generates DAG decomposition and agent assignments"""
    
    def __init__(self, 
                 obs_dim: int = 512,
                 hidden_dim: int = 256,
                 max_nodes: int = 10,
                 num_agents: int = 4):
        super().__init__()
        
        # Observation encoder (Transformer)
        self.obs_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=obs_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1
            ),
            num_layers=3
        )
        
        # Graph state encoder (GNN)
        self.graph_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.graph_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # DAG generation head
        self.dag_head = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, max_nodes * max_nodes)
        )
        
        # Agent assignment head
        self.agent_head = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, max_nodes * num_agents)
        )
        
        self.max_nodes = max_nodes
        self.num_agents = num_agents
        
    def forward(self, obs: torch.Tensor, graph_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate DAG and agent assignments"""
        
        # Encode observation
        obs_encoded = self.obs_encoder(obs.unsqueeze(1)).squeeze(1)
        
        # Encode graph state
        edge_index = self._build_edge_index(graph_state)
        x = self.graph_conv1(graph_state, edge_index)
        x = F.relu(x)
        graph_encoded = self.graph_conv2(x, edge_index)
        graph_encoded = graph_encoded.mean(dim=0)  # Global pooling
        
        # Combine features
        combined = torch.cat([obs_encoded, graph_encoded], dim=-1)
        
        # Generate DAG adjacency matrix
        adj_logits = self.dag_head(combined)
        adj_matrix = adj_logits.view(self.max_nodes, self.max_nodes)
        
        # Enforce DAG constraints (upper triangular)
        mask = torch.triu(torch.ones_like(adj_matrix), diagonal=1)
        adj_matrix = adj_matrix * mask
        adj_matrix = torch.sigmoid(adj_matrix)
        
        # Generate agent assignments
        agent_logits = self.agent_head(combined)
        agent_assignments = agent_logits.view(self.max_nodes, self.num_agents)
        agent_assignments = F.softmax(agent_assignments, dim=-1)
        
        return adj_matrix, agent_assignments
    
    def _build_edge_index(self, graph_state: torch.Tensor) -> torch.Tensor:
        """Build edge index from graph state"""
        # Implementation depends on graph representation
        pass
```

## 4. WebArena Integration

### 4.1 Enhanced WebArenaMAS
**File:** `mas/enhanced_webarena_mas.py`
```python
from typing import Dict, Any, List, Optional
import numpy as np
from algorithms.ppo_lagrangian import PPOLagrangian
from algorithms.p3o import P3O
from algorithms.macpo import MACPO
from models.orchestrator import OrchestratorPolicy

class EnhancedWebArenaMAS:
    """Research-grade MAS with multiple constraint methods"""
    
    def __init__(self, 
                 method: str = 'ppo_lagrangian',  # or 'p3o' or 'macpo'
                 budget: float = 1.0,
                 num_agents: int = 4):
        
        # Initialize algorithm
        if method == 'ppo_lagrangian':
            self.algorithm = PPOLagrangian(budget=budget)
        elif method == 'p3o':
            self.algorithm = P3O(budget=budget)
        elif method == 'macpo':
            self.algorithm = MACPO(num_agents=num_agents)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.method = method
        self.orchestrator = OrchestratorPolicy(num_agents=num_agents)
        
        # Agent pool (existing)
        self.agent_pool = self._initialize_agent_pool()
        
        # Metrics tracking
        self.metrics = {
            'success_rate': [],
            'cost_guarantee_rate': [],
            'avg_cost': [],
            'duality_gap': [] if method == 'ppo_lagrangian' else None,
            'dag_complexity': [],
            'communication_efficiency': []
        }
        
    def solve_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Solve WebArena task with cost guarantees"""
        
        # Encode task
        state = self._encode_state(task)
        
        # Generate DAG decomposition
        adj_matrix, agent_assignments = self.orchestrator(
            torch.FloatTensor(state['obs']),
            torch.FloatTensor(state['graph'])
        )
        
        # Build DAG
        dag = self._build_dag_from_adjacency(adj_matrix)
        
        # Execute DAG with assigned agents
        trajectory = self._execute_dag(dag, agent_assignments, task)
        
        # Calculate metrics
        success = self._evaluate_success(trajectory)
        cost = sum(t['cost'] for t in trajectory)
        
        # Update policy
        update_info = self.algorithm.update(trajectory)
        
        # Track metrics
        self._update_metrics(success, cost, dag, update_info)
        
        return {
            'success': success,
            'cost': cost,
            'trajectory': trajectory,
            'dag': dag,
            'method_info': update_info
        }
    
    def _build_dag_from_adjacency(self, adj_matrix: torch.Tensor) -> nx.DiGraph:
        """Convert adjacency matrix to DAG"""
        import networkx as nx
        
        G = nx.DiGraph()
        adj = adj_matrix.detach().cpu().numpy()
        
        n_nodes = adj.shape[0]
        for i in range(n_nodes):
            G.add_node(i)
            
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj[i, j] > 0.5:  # Threshold
                    G.add_edge(i, j, weight=adj[i, j])
        
        # Verify DAG property
        if not nx.is_directed_acyclic_graph(G):
            # Remove cycles if any (shouldn't happen with upper triangular)
            G = self._remove_cycles(G)
            
        return G
    
    def _execute_dag(self, dag: nx.DiGraph, 
                    agent_assignments: torch.Tensor,
                    task: Dict) -> List[Dict]:
        """Execute DAG with parallel/sequential scheduling"""
        
        trajectory = []
        
        # Topological sort for execution order
        exec_order = list(nx.topological_sort(dag))
        
        # Group by levels for parallel execution
        levels = self._compute_dag_levels(dag)
        
        for level_nodes in levels:
            if self._should_parallelize(level_nodes):
                # Execute in parallel
                results = self._execute_parallel(level_nodes, agent_assignments, task)
            else:
                # Execute sequentially
                results = self._execute_sequential(level_nodes, agent_assignments, task)
            
            trajectory.extend(results)
            
            # Check budget
            total_cost = sum(t['cost'] for t in trajectory)
            if total_cost > self.algorithm.budget * 1.05:
                break  # Hard stop at 105% budget
                
        return trajectory
    
    def _update_metrics(self, success: bool, cost: float, 
                       dag: nx.DiGraph, update_info: Dict):
        """Update research metrics"""
        
        self.metrics['success_rate'].append(success)
        self.metrics['cost_guarantee_rate'].append(cost <= self.algorithm.budget * 1.05)
        self.metrics['avg_cost'].append(cost)
        
        # DAG metrics
        self.metrics['dag_complexity'].append({
            'nodes': dag.number_of_nodes(),
            'edges': dag.number_of_edges(),
            'diameter': nx.diameter(dag) if nx.is_weakly_connected(dag) else -1,
            'avg_degree': np.mean([d for n, d in dag.degree()])
        })
        
        # Method-specific metrics
        if self.method == 'ppo_lagrangian' and 'duality_gap' in update_info:
            self.metrics['duality_gap'].append(update_info['duality_gap'])
```

## 5. Experiments & Evaluation

### 5.1 Main Experiment Script
**File:** `experiments/run_comparison.py`
```python
import json
import numpy as np
from typing import Dict, List
import wandb
from enhanced_webarena_mas import EnhancedWebArenaMAS

def run_experiment(method: str, 
                  tasks: List[Dict],
                  seeds: List[int] = [42, 1337, 2024]) -> Dict:
    """Run experiment for one method across seeds"""
    
    results = {
        'method': method,
        'seeds': {},
        'aggregated': {}
    }
    
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize MAS
        mas = EnhancedWebArenaMAS(method=method, budget=1.0)
        
        # Training loop
        seed_results = train_mas(mas, tasks)
        results['seeds'][seed] = seed_results
        
    # Aggregate across seeds
    results['aggregated'] = aggregate_results(results['seeds'])
    
    return results

def train_mas(mas: EnhancedWebArenaMAS, 
              tasks: List[Dict],
              episodes: int = 10000) -> Dict:
    """Training loop with evaluation"""
    
    train_tasks = tasks[:500]
    val_tasks = tasks[500:650]
    test_tasks = tasks[650:812]
    
    results = {
        'train_history': [],
        'val_history': [],
        'test_results': None
    }
    
    for episode in range(episodes):
        # Sample task
        task = np.random.choice(train_tasks)
        
        # Execute task
        outcome = mas.solve_task(task)
        
        # Log to wandb
        wandb.log({
            f'{mas.method}/success': outcome['success'],
            f'{mas.method}/cost': outcome['cost'],
            f'{mas.method}/duality_gap': outcome['method_info'].get('duality_gap', 0)
        })
        
        # Validation
        if episode % 100 == 0:
            val_metrics = evaluate(mas, val_tasks[:20])
            results['val_history'].append(val_metrics)
            
            print(f"Episode {episode}: "
                  f"Success={val_metrics['success_rate']:.2%}, "
                  f"Cost=${val_metrics['avg_cost']:.3f}, "
                  f"CGR={val_metrics['cost_guarantee_rate']:.2%}")
    
    # Final test evaluation
    results['test_results'] = evaluate(mas, test_tasks)
    
    return results

def evaluate(mas: EnhancedWebArenaMAS, 
            tasks: List[Dict]) -> Dict:
    """Evaluate on task set"""
    
    outcomes = []
    for task in tasks:
        outcome = mas.solve_task(task)
        outcomes.append(outcome)
    
    return {
        'success_rate': np.mean([o['success'] for o in outcomes]),
        'avg_cost': np.mean([o['cost'] for o in outcomes]),
        'cost_guarantee_rate': np.mean([o['cost'] <= 1.05 for o in outcomes]),
        'dag_complexity': np.mean([o['dag'].number_of_nodes() for o in outcomes])
    }

def main():
    """Main experiment comparing methods"""
    
    # Load WebArena tasks
    with open('data/webarena_tasks.json', 'r') as f:
        tasks = json.load(f)
    
    # Methods to compare
    methods = [
        'ppo_lagrangian',  # Dual method
        'p3o',             # Primal method
        'macpo',           # MARL baseline
        'single_llm',      # Single agent baseline
        'heuristic_mas'    # Rule-based MAS
    ]
    
    # Run experiments
    all_results = {}
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running {method}")
        print('='*50)
        
        results = run_experiment(method, tasks)
        all_results[method] = results
        
    # Generate tables and figures
    generate_results_table(all_results)
    generate_figures(all_results)
    
    # Save results
    with open('results/comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    wandb.init(project="mas-webarena-neurips")
    main()
```

### 5.2 Metrics & Visualization
**File:** `evaluation/metrics.py`
```python
class ResearchMetrics:
    """Calculate all research metrics"""
    
    @staticmethod
    def calculate_emergence_metrics(trajectories: List[Dict]) -> Dict:
        """Quantify emergent coordination"""
        
        # Communication entropy
        comm_entropy = calculate_communication_entropy(trajectories)
        
        # Agent diversity (Simpson's index)
        agent_usage = count_agent_usage(trajectories)
        diversity = 1 - sum((n/sum(agent_usage.values()))**2 
                           for n in agent_usage.values())
        
        # Parallelization efficiency
        parallel_steps = count_parallel_steps(trajectories)
        total_steps = len(trajectories)
        parallel_rate = parallel_steps / total_steps
        
        return {
            'communication_entropy': comm_entropy,
            'agent_diversity': diversity,
            'parallelization_rate': parallel_rate
        }
    
    @staticmethod
    def calculate_graph_metrics(dags: List[nx.DiGraph]) -> Dict:
        """DAG complexity metrics"""
        
        metrics = {
            'avg_nodes': np.mean([g.number_of_nodes() for g in dags]),
            'avg_edges': np.mean([g.number_of_edges() for g in dags]),
            'avg_diameter': np.mean([nx.diameter(g) if nx.is_weakly_connected(g) else -1 
                                    for g in dags]),
            'avg_clustering': np.mean([nx.average_clustering(g.to_undirected()) 
                                      for g in dags])
        }
        
        return metrics
```

## 6. Required Figures

### 6.1 Figure Generation
**File:** `visualization/figures.py`
```python
import matplotlib.pyplot as plt
import seaborn as sns

def generate_all_figures(results: Dict):
    """Generate all paper figures"""
    
    # Figure 1: Learning curves
    fig1 = plot_learning_curves(results)
    
    # Figure 2: Pareto frontier
    fig2 = plot_pareto_frontier(results)
    
    # Figure 3: Duality gap comparison
    fig3 = plot_duality_gap(results)
    
    # Figure 4: DAG evolution
    fig4 = plot_dag_evolution(results)
    
    # Figure 5: Cost guarantee satisfaction
    fig5 = plot_cost_guarantees(results)
    
    return [fig1, fig2, fig3, fig4, fig5]

def plot_learning_curves(results: Dict):
    """Success rate, cost, CGR over episodes"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for method, data in results.items():
        # Plot with confidence intervals
        episodes = range(len(data['val_history']))
        
        success_mean = [v['success_rate'] for v in data['val_history']]
        axes[0].plot(episodes, success_mean, label=method)
        
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Success Rate')
    axes[0].legend()
    
    return fig

def plot_pareto_frontier(results: Dict):
    """Success vs Cost trade-off"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for method, data in results.items():
        test_results = data['test_results']
        ax.scatter(test_results['avg_cost'], 
                  test_results['success_rate'],
                  s=100, label=method)
    
    ax.set_xlabel('Average Cost ($)')
    ax.set_ylabel('Success Rate')
    ax.legend()
    
    return fig
```

## 7. Ablation Studies

### 7.1 Ablation Runner
**File:** `experiments/ablations.py`
```python
def run_ablations():
    """Run all ablation studies"""
    
    ablations = {
        'full': {},  # Full system
        'no_memory': {'use_memory': False},
        'no_communication': {'enable_communication': False},
        'no_dag': {'use_dag': False, 'force_sequential': True},
        'no_cost_penalty': {'cost_weight': 0.0},
        'fixed_decomposition': {'use_learned_decomposition': False},
        'fixed_epsilon': {'adaptive_epsilon': False, 'epsilon': 0.1}
    }
    
    results = {}
    for name, config in ablations.items():
        mas = EnhancedWebArenaMAS(method='p3o', **config)
        results[name] = train_mas(mas, tasks)
    
    return results
```

## 8. Implementation Timeline

### Week 1: Core Algorithms
- [ ] Implement PPO-Lagrangian with duality gap tracking
- [ ] Implement P3O (primal method)
- [ ] Implement MACPO baseline
- [ ] Test convergence on simple tasks

### Week 2: Orchestrator & DAG
- [ ] Implement OrchestratorPolicy with Transformer+GNN
- [ ] Implement DAG generation and execution
- [ ] Test DAG constraints and scheduling
- [ ] Integrate with existing WebArena wrapper

### Week 3: Experiments
- [ ] Run main comparison (3 seeds, all methods)
- [ ] Run ablation studies
- [ ] Generate all figures
- [ ] Calculate statistical significance

### Week 4: Analysis & Writing
- [ ] Analyze emergent strategies
- [ ] Create results tables
- [ ] Write method section
- [ ] Prepare supplementary materials

## 9. File Structure
```
mas_webarena/
├── algorithms/
│   ├── ppo_lagrangian.py
│   ├── p3o.py
│   └── macpo.py
├── models/
│   ├── orchestrator.py
│   └── networks.py
├── mas/
│   ├── enhanced_webarena_mas.py
│   └── agent_pool.py
├── experiments/
│   ├── run_comparison.py
│   ├── ablations.py
│   └── configs/
├── evaluation/
│   ├── metrics.py
│   └── statistical_tests.py
├── visualization/
│   ├── figures.py
│   └── tables.py
├── data/
│   └── webarena_tasks.json
└── results/
    └── comparison_results.json
```

## 10. Execution Commands

```bash
# Install dependencies
pip install torch torch-geometric networkx wandb

# Run main experiments
python experiments/run_comparison.py \
    --methods ppo_lagrangian,p3o,macpo \
    --seeds 42,1337,2024 \
    --episodes 10000

# Run ablations
python experiments/ablations.py \
    --base_method p3o \
    --ablations all

# Generate figures
python visualization/figures.py \
    --results results/comparison_results.json \
    --output figures/

# Quick test (1 hour)
python experiments/run_comparison.py \
    --methods p3o \
    --episodes 100 \
    --quick_test
```

## 11. Expected Results Table

| Method | Success↑ | Cost↓ | CGR↑ | Duality Gap | Time(s)↓ |
|--------|----------|-------|------|-------------|----------|
| Single-LLM | 0.65±0.03 | 0.82±0.05 | 0.45±0.04 | - | 8.3±1.2 |
| Heuristic-MAS | 0.68±0.02 | 0.75±0.04 | 0.62±0.03 | - | 7.1±0.9 |
| MACPO | 0.74±0.03 | 0.70±0.04 | 0.88±0.03 | - | 6.5±0.8 |
| PPO-Lagrangian | 0.76±0.02 | 0.68±0.04 | 0.92±0.02 | 0.12±0.03 | 6.8±0.7 |
| **P3O (Ours)** | **0.77±0.02** | **0.66±0.03** | **0.95±0.01** | **-** | **6.6±0.7** |

## 12. Key Contributions Summary

1. **First CMDP formulation** for multi-agent web automation
2. **Novel end-to-end learned DAG decomposition** 
3. **Empirical comparison** showing primal methods (P3O) outperform dual (Lagrangian) in cooperative MAS
4. **HRL framework** with learned temporal abstraction
5. **95% cost guarantee** with maintained performance

## Critical Success Factors

- **Focus**: Implement P3O first (simpler, better results)
- **Baseline**: MACPO is crucial for credibility
- **Metrics**: Duality gap comparison is the key insight
- **Framing**: Emphasize HRL perspective throughout

This plan synthesizes all feedback and provides a complete implementation path for NeurIPS-level research.