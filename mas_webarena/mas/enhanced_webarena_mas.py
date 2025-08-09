import torch
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.ppo_lagrangian import PPOLagrangian
from algorithms.p3o import P3O
from algorithms.macpo import MACPO
from models.orchestrator import OrchestratorPolicy

# Import existing WebArena components if available
try:
    # Try to import from existing WebArena MAS implementation
    from run_webarena_experiment import WebArenaMAS, Agent
    from WebArenaSpecificMetrics import WebArenaMetrics
    HAS_WEBARENA = True
except ImportError:
    HAS_WEBARENA = False
    print("Warning: Existing WebArena components not found, using mock implementations")


class MockAgent:
    """Mock agent for when WebArena components are not available"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.capabilities = ['click', 'type', 'navigate']
    
    def execute_action(self, action: Dict) -> Dict:
        # Mock execution with random success/cost
        return {
            'success': np.random.random() > 0.3,
            'cost': np.random.uniform(0.01, 0.1),
            'reward': np.random.uniform(0, 1) if np.random.random() > 0.3 else 0
        }


class EnhancedWebArenaMAS:
    """Research-grade MAS with multiple constraint methods"""
    
    def __init__(self, 
                 method: str = 'ppo_lagrangian',  # or 'p3o' or 'macpo'
                 budget: float = 1.0,
                 num_agents: int = 4,
                 state_dim: int = 128,
                 action_dim: int = 64,
                 max_nodes: int = 10,
                 device: str = 'cpu',
                 use_orchestrator: bool = True,
                 parallel_execution: bool = True):
        
        self.method = method
        self.budget = budget
        self.num_agents = num_agents
        self.device = torch.device(device)
        self.use_orchestrator = use_orchestrator
        self.parallel_execution = parallel_execution
        
        # Initialize algorithm
        if method == 'ppo_lagrangian':
            self.algorithm = PPOLagrangian(
                state_dim=state_dim, 
                action_dim=action_dim, 
                budget=budget
            )
        elif method == 'p3o':
            self.algorithm = P3O(
                state_dim=state_dim, 
                action_dim=action_dim, 
                budget=budget
            )
        elif method == 'macpo':
            self.algorithm = MACPO(
                num_agents=num_agents,
                state_dim=state_dim, 
                action_dim=action_dim//num_agents,  # Distribute action space
                budget=budget
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Orchestrator for hierarchical control
        if self.use_orchestrator:
            self.orchestrator = OrchestratorPolicy(
                obs_dim=state_dim,
                max_nodes=max_nodes,
                num_agents=num_agents
            ).to(self.device)
        else:
            self.orchestrator = None
        
        # Agent pool (use existing if available, otherwise mock)
        self.agent_pool = self._initialize_agent_pool()
        
        # Metrics tracking
        self.metrics = {
            'success_rate': [],
            'cost_guarantee_rate': [],
            'avg_cost': [],
            'avg_reward': [],
            'duality_gap': [] if method == 'ppo_lagrangian' else None,
            'dag_complexity': [],
            'communication_efficiency': [],
            'episode_lengths': [],
            'constraint_violations': []
        }
        
        # Episode counter
        self.episode_count = 0
        
    def _initialize_agent_pool(self) -> List:
        """Initialize agent pool"""
        if HAS_WEBARENA:
            # Use existing WebArena agents
            agents = []
            for i in range(self.num_agents):
                agent = Agent(agent_id=f"agent_{i}")
                agents.append(agent)
            return agents
        else:
            # Use mock agents
            return [MockAgent(f"agent_{i}") for i in range(self.num_agents)]
    
    def solve_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Solve WebArena task with cost guarantees"""
        
        self.episode_count += 1
        
        # Encode task state
        state = self._encode_state(task)
        state_tensor = torch.FloatTensor(state['obs']).unsqueeze(0).to(self.device)
        
        # Generate DAG decomposition if using orchestrator
        if self.use_orchestrator:
            with torch.no_grad():
                graph_state = torch.FloatTensor(state.get('graph', np.zeros((5, 128)))).unsqueeze(0).to(self.device)
                adj_matrix, agent_assignments, node_difficulties = self.orchestrator(state_tensor, graph_state)
            
            # Build DAG from adjacency matrix
            dag = self.orchestrator.sample_dag(adj_matrix.squeeze(0))
            agent_assign = agent_assignments.squeeze(0).cpu().numpy()
            difficulties = node_difficulties.squeeze(0).cpu().numpy()
        else:
            # Use simple sequential decomposition
            dag = self._create_simple_dag(task)
            agent_assign = np.ones((len(dag.nodes()), self.num_agents)) / self.num_agents
            difficulties = np.ones(len(dag.nodes())) * 0.5
        
        # Execute DAG with assigned agents
        trajectory = self._execute_dag(dag, agent_assign, difficulties, task, state_tensor)
        
        # Calculate metrics
        success = self._evaluate_success(trajectory, task)
        total_cost = sum(t['cost'] for t in trajectory)
        total_reward = sum(t['reward'] for t in trajectory)
        
        # Update policy based on method
        if self.method == 'macpo':
            # Group trajectory by agent for MACPO
            trajectories_by_agent = self._group_trajectory_by_agent(trajectory)
            update_info = self.algorithm.update(trajectories_by_agent)
        else:
            # Single trajectory for PPO-Lagrangian and P3O
            update_info = self.algorithm.update(trajectory)
        
        # Track metrics
        self._update_metrics(success, total_cost, total_reward, dag, update_info, trajectory)
        
        result = {
            'success': success,
            'cost': total_cost,
            'reward': total_reward,
            'trajectory': trajectory,
            'dag': dag,
            'method_info': update_info,
            'episode': self.episode_count,
            'dag_metrics': self.orchestrator.compute_dag_metrics(dag) if self.use_orchestrator else {}
        }
        
        return result
    
    def _encode_state(self, task: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Encode task into state representation"""
        # Mock state encoding - in practice this would use actual WebArena state
        obs_dim = 128
        
        # Create observation from task
        task_text = task.get('intent', '') + ' ' + str(task.get('sites', []))
        
        # Simple text encoding (hash-based for reproducibility)
        hash_val = hash(task_text)
        obs = np.random.RandomState(hash_val % 2**31).random(obs_dim).astype(np.float32)
        
        # Graph state (representing current DOM structure or site state)
        graph_state = np.random.RandomState((hash_val + 1) % 2**31).random((5, 128)).astype(np.float32)
        
        return {
            'obs': obs,
            'graph': graph_state,
            'task_id': task.get('task_id', 'unknown')
        }
    
    def _create_simple_dag(self, task: Dict[str, Any]) -> nx.DiGraph:
        """Create a simple sequential DAG for non-orchestrator mode"""
        G = nx.DiGraph()
        
        # Simple 3-step process: navigate -> interact -> verify
        steps = ['navigate', 'interact', 'verify']
        
        for i, step in enumerate(steps):
            G.add_node(i, action=step)
            if i > 0:
                G.add_edge(i-1, i)
        
        return G
    
    def _execute_dag(self, dag: nx.DiGraph, 
                    agent_assignments: np.ndarray,
                    node_difficulties: np.ndarray,
                    task: Dict, 
                    state_tensor: torch.Tensor) -> List[Dict]:
        """Execute DAG with parallel/sequential scheduling"""
        
        trajectory = []
        
        if len(dag.nodes()) == 0:
            # Empty DAG, return empty trajectory
            return trajectory
        
        # Topological sort for execution order
        try:
            exec_order = list(nx.topological_sort(dag))
        except nx.NetworkXError:
            # If not a DAG, use node order
            exec_order = list(dag.nodes())
        
        # Group by levels for parallel execution
        if self.parallel_execution and len(exec_order) > 1:
            levels = self._compute_dag_levels(dag)
        else:
            levels = [[node] for node in exec_order]
        
        current_state = state_tensor
        total_cost = 0
        
        for level_idx, level_nodes in enumerate(levels):
            if self._should_parallelize(level_nodes) and self.parallel_execution:
                # Execute in parallel
                results = self._execute_parallel(level_nodes, agent_assignments, node_difficulties, task, current_state)
            else:
                # Execute sequentially
                results = self._execute_sequential(level_nodes, agent_assignments, node_difficulties, task, current_state)
            
            # Add results to trajectory
            for result in results:
                # Add state information to trajectory
                result['state'] = current_state.squeeze(0).cpu().numpy()
                result['level'] = level_idx
                trajectory.append(result)
                total_cost += result['cost']
            
            # Check budget constraint
            if total_cost > self.budget * 1.05:  # Hard stop at 105% budget (matches CLAUDE.md)
                break
        
        return trajectory
    
    def _compute_dag_levels(self, dag: nx.DiGraph) -> List[List[int]]:
        """Compute levels of DAG for parallel execution"""
        levels = []
        remaining_nodes = set(dag.nodes())
        
        while remaining_nodes:
            # Find nodes with no incoming edges from remaining nodes
            current_level = []
            for node in remaining_nodes:
                predecessors = set(dag.predecessors(node)) & remaining_nodes
                if not predecessors:
                    current_level.append(node)
            
            if not current_level:
                # Circular dependency, break it
                current_level = [remaining_nodes.pop()]
            
            levels.append(current_level)
            remaining_nodes -= set(current_level)
        
        return levels
    
    def _should_parallelize(self, level_nodes: List[int]) -> bool:
        """Decide whether to parallelize execution of level nodes"""
        return len(level_nodes) > 1 and len(level_nodes) <= self.num_agents
    
    def _execute_parallel(self, level_nodes: List[int], 
                         agent_assignments: np.ndarray,
                         node_difficulties: np.ndarray,
                         task: Dict, 
                         state: torch.Tensor) -> List[Dict]:
        """Execute nodes in parallel"""
        results = []
        
        for node_id in level_nodes:
            # Choose agent based on assignment probabilities
            if node_id < len(agent_assignments):
                agent_probs = agent_assignments[node_id]
                agent_id = np.random.choice(self.num_agents, p=agent_probs)
            else:
                agent_id = np.random.choice(self.num_agents)
            
            # Get agent action based on method
            action, log_prob = self._get_agent_action(agent_id, state, node_id)
            
            # Execute action with assigned agent
            agent = self.agent_pool[agent_id]
            execution_result = agent.execute_action({'action': action.item(), 'node_id': node_id})
            
            # Compute cost based on difficulty and method
            base_cost = np.random.uniform(0.01, 0.1)
            if node_id < len(node_difficulties):
                difficulty_multiplier = 1 + node_difficulties[node_id]
            else:
                difficulty_multiplier = 1.5
            
            cost = base_cost * difficulty_multiplier
            
            results.append({
                'action': action.item(),
                'agent_id': agent_id,
                'node_id': node_id,
                'cost': cost,
                'reward': execution_result['reward'],
                'success': execution_result['success'],
                'log_prob': log_prob.item(),
                'done': False
            })
        
        return results
    
    def _execute_sequential(self, level_nodes: List[int], 
                           agent_assignments: np.ndarray,
                           node_difficulties: np.ndarray,
                           task: Dict, 
                           state: torch.Tensor) -> List[Dict]:
        """Execute nodes sequentially"""
        # For sequential execution, we can reuse the parallel logic
        # but execute one at a time
        return self._execute_parallel(level_nodes, agent_assignments, node_difficulties, task, state)
    
    def _get_agent_action(self, agent_id: int, state: torch.Tensor, node_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from agent based on method"""
        if self.method == 'macpo':
            return self.algorithm.get_action(agent_id, state.squeeze(0))
        else:
            # For PPO-Lagrangian and P3O, use single policy
            return self.algorithm.get_action(state.squeeze(0))
    
    def _group_trajectory_by_agent(self, trajectory: List[Dict]) -> List[List[Dict]]:
        """Group trajectory by agent for MACPO"""
        agent_trajectories = [[] for _ in range(self.num_agents)]
        
        for step in trajectory:
            agent_id = step['agent_id']
            agent_trajectories[agent_id].append(step)
        
        return agent_trajectories
    
    def _evaluate_success(self, trajectory: List[Dict], task: Dict) -> bool:
        """Evaluate task success"""
        if not trajectory:
            return False
        
        # Simple success heuristic: at least 80% of actions succeeded
        success_count = sum(1 for t in trajectory if t['success'])
        success_rate = success_count / len(trajectory)
        
        return success_rate >= 0.8
    
    def _update_metrics(self, success: bool, cost: float, reward: float,
                       dag: nx.DiGraph, update_info: Dict, trajectory: List[Dict]):
        """Update research metrics"""
        
        self.metrics['success_rate'].append(success)
        self.metrics['cost_guarantee_rate'].append(cost <= self.budget * 1.05)
        self.metrics['avg_cost'].append(cost)
        self.metrics['avg_reward'].append(reward)
        self.metrics['episode_lengths'].append(len(trajectory))
        self.metrics['constraint_violations'].append(max(0, cost - self.budget * 1.05))
        
        # DAG metrics
        if self.use_orchestrator:
            dag_metrics = self.orchestrator.compute_dag_metrics(dag)
            self.metrics['dag_complexity'].append(dag_metrics)
        
        # Method-specific metrics
        if self.method == 'ppo_lagrangian' and 'duality_gap' in update_info:
            self.metrics['duality_gap'].append(update_info['duality_gap'])
    
    def get_metrics_summary(self, window: int = 100) -> Dict[str, float]:
        """Get summary of recent metrics"""
        if not self.metrics['success_rate']:
            return {}
        
        recent_slice = slice(-window, None) if len(self.metrics['success_rate']) > window else slice(None)
        
        summary = {
            'success_rate': np.mean(self.metrics['success_rate'][recent_slice]),
            'cost_guarantee_rate': np.mean(self.metrics['cost_guarantee_rate'][recent_slice]),
            'avg_cost': np.mean(self.metrics['avg_cost'][recent_slice]),
            'avg_reward': np.mean(self.metrics['avg_reward'][recent_slice]),
            'avg_episode_length': np.mean(self.metrics['episode_lengths'][recent_slice]),
            'constraint_violation_rate': np.mean([v > 0 for v in self.metrics['constraint_violations'][recent_slice]]),
            'episodes': len(self.metrics['success_rate'])
        }
        
        # Add method-specific metrics
        if self.method == 'ppo_lagrangian' and self.metrics['duality_gap']:
            summary['avg_duality_gap'] = np.mean(self.metrics['duality_gap'][recent_slice])
        
        # Add DAG metrics if available
        if self.metrics['dag_complexity'] and self.metrics['dag_complexity'][0]:
            dag_metrics = self.metrics['dag_complexity'][recent_slice]
            if dag_metrics:
                summary['avg_dag_nodes'] = np.mean([d['nodes'] for d in dag_metrics])
                summary['avg_dag_edges'] = np.mean([d['edges'] for d in dag_metrics])
        
        return summary
    
    def save_checkpoint(self, filepath: str):
        """Save complete system checkpoint"""
        checkpoint = {
            'method': self.method,
            'episode_count': self.episode_count,
            'metrics': self.metrics,
            'algorithm_state': None,
            'orchestrator_state': None
        }
        
        # Save algorithm state
        if hasattr(self.algorithm, 'save_checkpoint'):
            algorithm_path = filepath.replace('.pt', '_algorithm.pt')
            self.algorithm.save_checkpoint(algorithm_path)
            checkpoint['algorithm_path'] = algorithm_path
        
        # Save orchestrator state
        if self.use_orchestrator:
            checkpoint['orchestrator_state'] = self.orchestrator.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load complete system checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.episode_count = checkpoint['episode_count']
        self.metrics = checkpoint['metrics']
        
        # Load algorithm state
        if 'algorithm_path' in checkpoint and hasattr(self.algorithm, 'load_checkpoint'):
            self.algorithm.load_checkpoint(checkpoint['algorithm_path'])
        
        # Load orchestrator state
        if 'orchestrator_state' in checkpoint and self.use_orchestrator:
            self.orchestrator.load_state_dict(checkpoint['orchestrator_state'])
        
        print(f"Checkpoint loaded from {filepath}")


# Utility functions for integration with existing WebArena
def create_enhanced_mas_from_config(config_path: str) -> EnhancedWebArenaMAS:
    """Create EnhancedWebArenaMAS from configuration file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return EnhancedWebArenaMAS(**config)


def run_single_task_test(method: str = 'p3o') -> Dict:
    """Run a single task test for debugging"""
    # Create test task
    test_task = {
        'task_id': 'test_001',
        'intent': 'Navigate to shopping page and add item to cart',
        'sites': ['shopping.com']
    }
    
    # Create MAS
    mas = EnhancedWebArenaMAS(method=method, budget=0.5, num_agents=2)
    
    # Solve task
    result = mas.solve_task(test_task)
    
    print(f"Test Result - Method: {method}")
    print(f"Success: {result['success']}")
    print(f"Cost: {result['cost']:.3f} (Budget: {mas.budget})")
    print(f"Reward: {result['reward']:.3f}")
    print(f"DAG nodes: {result['dag_metrics'].get('nodes', 0)}")
    
    return result


if __name__ == "__main__":
    # Run test
    result = run_single_task_test('p3o')