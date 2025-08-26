import torch
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
import json
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.ppo_lagrangian import PPOLagrangian
from algorithms.p3o import P3O
from algorithms.macpo import MACPO
# Legacy neural orchestrator (deprecated in favor of LLM orchestrator)
# from models.orchestrator import OrchestratorPolicy
from orchestrator import (
    LLMOrchestratorPolicy, 
    DAGCacheManager, 
    ReplanningEngine,
    build_context,
    update_context_with_result
)
from utils.budget_tracker import BudgetTracker

# Import WebArenaMetrics for enhanced tracking
try:
    from WebArenaMetrics import WebArenaMetrics
    HAS_WEBARENA_METRICS = True
except ImportError:
    HAS_WEBARENA_METRICS = False
    print("Warning: WebArenaMetrics not found, using basic metrics")

# Import existing WebArena components if available
try:
    # Try to import from existing WebArena MAS implementation
    from run_webarena_experiment import WebArenaMAS, Agent
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
                 method: str = 'p3o',  # 'ppo_lagrangian', 'p3o', or 'macpo'
                 budget: float = 1.0,
                 num_agents: int = 4,
                 state_dim: int = 128,
                 action_dim: int = 64,
                 max_nodes: int = 10,
                 device: str = 'cpu',
                 use_llm_orchestrator: bool = True,  # Primary orchestration mode
                 llm_model: str = "gpt-4-turbo",
                 enable_replanning: bool = True):
        
        self.method = method
        self.budget = budget
        self.num_agents = num_agents
        self.device = torch.device(device)
        self.use_llm_orchestrator = use_llm_orchestrator
        self.enable_replanning = enable_replanning
        
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
        
        # Neural orchestrator (deprecated - use LLM orchestrator instead)
        self.orchestrator = None
        
        # LLM-based orchestrator components
        if self.use_llm_orchestrator:
            self.llm_orchestrator = LLMOrchestratorPolicy(
                llm_model=llm_model,
                method=method,
                budget=budget,
                max_nodes=max_nodes,
                num_agents=num_agents,
                state_dim=state_dim,
                action_dim=action_dim,
                device=device
            )
            
            # Cache manager for successful DAGs
            self.cache_manager = DAGCacheManager(
                cache_dir=f"dag_cache_{method}",
                max_cache_size=1000
            )
            
            # Replanning engine
            if self.enable_replanning:
                self.replanning_engine = ReplanningEngine(
                    orchestrator=self.llm_orchestrator,
                    replan_threshold=0.3,
                    max_replans_per_task=3
                )
            else:
                self.replanning_engine = None
        else:
            self.llm_orchestrator = None
            self.cache_manager = None
            self.replanning_engine = None
        
        # Budget tracker
        self.budget_tracker = BudgetTracker(initial_budget=budget)
        
        # Enhanced metrics tracking
        if HAS_WEBARENA_METRICS:
            self.webarena_metrics = WebArenaMetrics()
        else:
            self.webarena_metrics = None
        
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
    
    def _json_dag_to_networkx(self, dag_json: Dict) -> nx.DiGraph:
        """Convert JSON DAG format to NetworkX DiGraph"""
        G = nx.DiGraph()
        
        # Add nodes
        for node in dag_json.get('nodes', []):
            node_id = node['id']
            G.add_node(node_id, **node)
        
        # Add edges
        for edge in dag_json.get('edges', []):
            if len(edge) == 2:
                G.add_edge(edge[0], edge[1])
        
        return G
    
    def _find_executable_nodes(self, dag: nx.DiGraph, executed: set) -> List[int]:
        """Find nodes that can be executed (dependencies satisfied)"""
        executable = []
        
        for node_id in dag.nodes():
            if node_id in executed:
                continue
                
            # Check if all dependencies are satisfied
            dependencies_met = True
            for pred in dag.predecessors(node_id):
                if pred not in executed:
                    dependencies_met = False
                    break
                    
            if dependencies_met:
                executable.append(node_id)
                
        return executable
    
    def _execute_node(self, node_id: int, dag: Dict, task: Dict, context: Dict) -> Dict:
        """Execute a single node"""
        # Get node info
        if isinstance(dag, dict) and 'nodes' in dag:
            # JSON DAG format
            node_info = None
            for node in dag['nodes']:
                if node['id'] == node_id:
                    node_info = node
                    break
            if not node_info:
                node_info = {'id': node_id, 'task': f'action_{node_id}', 'estimated_cost': 0.05}
        else:
            # NetworkX format
            node_info = dag.nodes[node_id] if node_id in dag.nodes else {'id': node_id, 'task': f'action_{node_id}'}
        
        # Get assigned agent (from JSON DAG)
        if isinstance(dag, dict) and 'agent_assignments' in dag:
            agent_name = dag['agent_assignments'].get(str(node_id), 'gpt-4-turbo')
            # Map agent name to index (simplified)
            agent_mapping = {'claude-3.5': 0, 'gpt-4-turbo': 1, 'kimi-k2': 2, 'gemini-1.5': 3}
            agent_id = agent_mapping.get(agent_name, 0) % len(self.agent_pool)
        else:
            agent_id = np.random.choice(len(self.agent_pool))
        
        # Execute action (mock for now - would call actual WebArena agent)
        success = np.random.random() > 0.2  # 80% success rate
        
        result = {
            'node_id': node_id,
            'action': node_info.get('task', f'action_{node_id}'),
            'agent_id': agent_id,
            'cost': node_info.get('estimated_cost', 0.05),
            'reward': node_info.get('expected_reward', 0.1) if success else 0,
            'success': success,
            'observation': f"Executed {node_info.get('task', f'action_{node_id}')}",
            'timestamp': time.time()
        }
        
        return result
    
    def _is_task_complete(self, trajectory: List[Dict], task: Dict) -> bool:
        """Check if task is complete"""
        if not trajectory:
            return False
        
        # Simple completion check based on success rate and trajectory length
        expected_steps = task.get('expected_steps', 5)
        if len(trajectory) >= expected_steps:
            success_count = sum(1 for t in trajectory if t.get('success', False))
            return success_count >= expected_steps * 0.6  # 60% success rate threshold
        
        return False
    
    def _calculate_reward(self, trajectory: List[Dict], success: bool, total_cost: float) -> float:
        """Calculate final reward based on trajectory"""
        if not trajectory:
            return 0.0
        
        # Base reward from trajectory
        trajectory_reward = sum(t.get('reward', 0) for t in trajectory)
        
        # Success bonus
        success_bonus = 1.0 if success else 0.0
        
        # Cost penalty (encourage efficient solutions)
        cost_penalty = max(0, total_cost - self.budget) * 2.0
        
        return trajectory_reward + success_bonus - cost_penalty
    
    def _prepare_trajectory_for_rl(self, trajectory: List[Dict], context: Dict, dag: Dict, logits: torch.Tensor) -> List[Dict]:
        """Prepare trajectory for RL update with LLM integration"""
        rl_trajectory = []
        
        for step in trajectory:
            # Convert to RL format
            rl_step = {
                'state': step.get('state', np.zeros(128)),
                'action': step.get('action', 0),
                'reward': step.get('reward', 0),
                'cost': step.get('cost', 0),
                'done': step.get('done', False),
                'log_prob': step.get('log_prob', 0),
                'success': step.get('success', False)
            }
            
            # Add LLM-specific information
            if 'node_id' in step:
                rl_step['node_id'] = step['node_id']
            if 'agent_id' in step:
                rl_step['agent_id'] = step['agent_id']
            
            rl_trajectory.append(rl_step)
        
        return rl_trajectory
    
    def _compute_dag_metrics(self, dag: nx.DiGraph) -> Dict:
        """Compute DAG metrics"""
        if len(dag.nodes()) == 0:
            return {'nodes': 0, 'edges': 0, 'diameter': 0, 'avg_degree': 0}
        
        metrics = {
            'nodes': dag.number_of_nodes(),
            'edges': dag.number_of_edges(),
            'avg_degree': np.mean([d for n, d in dag.degree()]) if dag.degree() else 0
        }
        
        if nx.is_weakly_connected(dag):
            metrics['diameter'] = nx.diameter(dag.to_undirected())
        else:
            metrics['diameter'] = -1  # Disconnected
            
        return metrics
    
    def solve_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete implementation with LLM orchestrator
        """
        self.episode_count += 1
        
        # Initialize components
        trajectory = []
        total_cost = 0
        total_reward = 0
        replanning_count = 0
        
        # Reset budget tracker for this task
        self.budget_tracker.reset()
        
        # Build initial context and generate DAG
        if self.use_llm_orchestrator:
            # LLM-based orchestration (primary mode)
            context = build_context(
                task=task,
                current_observation=task.get('observation', ''),
                trajectory=[],
                budget_tracker=self.budget_tracker,
                episode=self.episode_count,
                method=self.method
            )
            
            # Generate initial DAG using LLM
            dag_gen_start = time.time()
            dag_json, logits = self.llm_orchestrator.generate_dag(
                context=context,
                use_cache=True,
                cache_manager=self.cache_manager
            )
            dag_gen_time = time.time() - dag_gen_start
            
            # Track DAG generation metrics
            if self.webarena_metrics:
                self.webarena_metrics.track_dag_generation(
                    dag_id=f"dag_{self.episode_count}_0",
                    generation_method="llm",
                    llm_model=self.llm_orchestrator.llm_model,
                    node_count=len(dag_json['dag'].get('nodes', [])),
                    edge_count=len(dag_json['dag'].get('edges', [])),
                    confidence_score=dag_json.get('confidence', 0.5),
                    generation_time=dag_gen_time,
                    generation_cost=0.01,  # Estimated cost - could be tracked by LLM orchestrator
                    validation_passed=True,  # Assume validation passed if no exception
                    cache_hit=False  # Could be tracked by cache manager
                )
            
            # Convert JSON DAG to NetworkX format for execution
            current_dag = dag_json
            dag = self._json_dag_to_networkx(dag_json['dag'])
            
        else:
            # Fallback mode: simple sequential decomposition
            dag = self._create_simple_dag(task)
            current_dag = {'dag': {'nodes': [{'id': i, 'task': f'step_{i}'} for i in dag.nodes()], 'confidence': 0.6}}
            logits = torch.zeros(1, 256)
            context = {}
        
        # Main execution loop
        executed_nodes = set()
        max_iterations = 30  # Prevent infinite loops
        
        for iteration in range(max_iterations):
            # Check if task is complete
            if self._is_task_complete(trajectory, task):
                break
                
            # Execute next batch of nodes
            executable_nodes = self._find_executable_nodes(dag, executed_nodes)
            
            if not executable_nodes:
                break  # No more nodes to execute
                
            # Execute nodes (possibly in parallel)
            batch_results = []
            for node_id in executable_nodes:
                exec_start = time.time()
                result = self._execute_node(
                    node_id=node_id,
                    dag=current_dag['dag'] if self.use_llm_orchestrator else dag,
                    task=task,
                    context=context if self.use_llm_orchestrator else {}
                )
                exec_time = time.time() - exec_start
                
                # Track action execution with enhanced metrics
                if self.webarena_metrics:
                    self.webarena_metrics.track_action(
                        agent_id=result.get('agent_id', f'agent_{node_id % self.num_agents}'),
                        action_type=result.get('action_type', 'web_action'),
                        success=result.get('success', False),
                        execution_time=exec_time,
                        cost=result.get('cost', 0.0),
                        error_message=result.get('error_message'),
                        node_id=node_id,
                        dag_confidence=current_dag.get('confidence', 0.5) if self.use_llm_orchestrator else None,
                        replanned=result.get('replanned', False),
                        llm_tokens_used=result.get('llm_tokens_used')
                    )
                
                batch_results.append(result)
                executed_nodes.add(node_id)
                
            # Update trajectory and costs
            trajectory.extend(batch_results)
            batch_cost = sum(r['cost'] for r in batch_results)
            total_cost += batch_cost
            
            # Update budget tracker
            if not self.budget_tracker.consume(batch_cost):
                break  # Budget exhausted
                
            # Check for re-planning (LLM orchestrator only)
            if self.use_llm_orchestrator and self.enable_replanning and self.replanning_engine:
                # Update context with latest state
                if batch_results:
                    context = update_context_with_result(context, batch_results[-1])
                
                should_replan, reason = self.replanning_engine.should_replan(
                    current_state=context,
                    trajectory=trajectory,
                    current_dag=current_dag
                )
                
                if should_replan:
                    replan_start = time.time()
                    new_dag_id = f"dag_{self.episode_count}_{replanning_count + 1}"
                    current_dag = self.replanning_engine.execute_replanning(
                        current_dag=current_dag,
                        current_state=context,
                        reason=reason,
                        completed_nodes=list(executed_nodes)
                    )
                    replan_time = time.time() - replan_start
                    
                    # Track replanning metrics
                    if self.webarena_metrics:
                        self.webarena_metrics.track_replanning(
                            original_dag_id=f"dag_{self.episode_count}_{replanning_count}",
                            new_dag_id=new_dag_id,
                            trigger_reason=reason,
                            completed_nodes=list(executed_nodes),
                            remaining_budget=self.budget_tracker.remaining_budget,
                            replan_generation_time=replan_time,
                            replan_cost=0.005  # Estimated replanning cost
                        )
                    
                    # Update NetworkX DAG
                    dag = self._json_dag_to_networkx(current_dag['dag'])
                    replanning_count += 1
                    
                    # Mark in trajectory
                    if trajectory:
                        trajectory[-1]['replanned'] = True
                        trajectory[-1]['replan_reason'] = reason
        
        # Calculate final metrics
        success = self._evaluate_success(trajectory, task)
        final_reward = self._calculate_reward(trajectory, success, total_cost)
        total_reward = final_reward
        
        # Prepare trajectory for RL update
        if self.use_llm_orchestrator:
            rl_trajectory = self._prepare_trajectory_for_rl(
                trajectory, context, current_dag, logits
            )
        else:
            rl_trajectory = trajectory
        
        # Update policy
        if self.method == 'macpo':
            trajectories_by_agent = self._group_trajectory_by_agent(rl_trajectory)
            update_info = self.algorithm.update(trajectories_by_agent)
        else:
            update_info = self.algorithm.update(rl_trajectory)
        
        # Cache successful DAG (LLM orchestrator only)
        if success and self.use_llm_orchestrator and self.cache_manager:
            self.cache_manager.cache_successful_dag(
                context=context,
                dag=current_dag,
                metrics={'success': success, 'cost': total_cost, 'reward': final_reward}
            )
        
        # Update metrics
        self._update_metrics(
            success=success,
            cost=total_cost,
            reward=final_reward,
            dag=dag,
            update_info=update_info,
            trajectory=trajectory,
            replanning_count=replanning_count
        )
        
        result = {
            'success': success,
            'cost': total_cost,
            'reward': final_reward,
            'trajectory': trajectory,
            'dag': current_dag if self.use_llm_orchestrator else dag,
            'method_info': update_info,
            'episode': self.episode_count,
            'replanning_count': replanning_count,
            'dag_metrics': self._compute_dag_metrics(dag)
        }
        
        if self.use_llm_orchestrator:
            result['logits'] = logits
        
        # Add enhanced metrics summary if available
        if self.webarena_metrics:
            result['enhanced_metrics'] = {
                'task_summary': self.webarena_metrics.get_task_summary(),
                'llm_orchestrator_metrics': self.webarena_metrics.get_llm_orchestrator_metrics(),
                'cost_analysis': self.webarena_metrics.get_cost_analysis()
            }
        
        return result
    
    def _create_simple_dag(self, task: Dict[str, Any]) -> nx.DiGraph:
        """Create a simple sequential DAG for fallback mode"""
        G = nx.DiGraph()
        
        # Simple 3-step process: navigate -> interact -> verify
        steps = ['navigate', 'interact', 'verify']
        
        for i, step in enumerate(steps):
            G.add_node(i, action=step, task=step)
            if i > 0:
                G.add_edge(i-1, i)
        
        return G
    
    
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
                       dag: nx.DiGraph, update_info: Dict, trajectory: List[Dict],
                       replanning_count: int = 0):
        """Update research metrics"""
        
        self.metrics['success_rate'].append(success)
        self.metrics['cost_guarantee_rate'].append(cost <= self.budget * 1.05)
        self.metrics['avg_cost'].append(cost)
        self.metrics['avg_reward'].append(reward)
        self.metrics['episode_lengths'].append(len(trajectory))
        self.metrics['constraint_violations'].append(max(0, cost - self.budget * 1.05))
        
        # Add replanning metrics
        if 'replanning_count' not in self.metrics:
            self.metrics['replanning_count'] = []
        self.metrics['replanning_count'].append(replanning_count)
        
        # DAG metrics
        if self.use_llm_orchestrator:
            dag_metrics = self._compute_dag_metrics(dag)
            self.metrics['dag_complexity'].append(dag_metrics)
        
        # Method-specific metrics
        if self.method == 'ppo_lagrangian' and 'duality_gap' in update_info:
            self.metrics['duality_gap'].append(update_info['duality_gap'])
    
    def get_metrics_summary(self, window: int = 100) -> Dict[str, float]:
        """Get summary of recent metrics"""
        if not self.metrics['success_rate']:
            return {}
        
        # Get recent metrics within window
        recent_success = self.metrics['success_rate'][-window:]
        recent_cost = self.metrics['avg_cost'][-window:]
        recent_reward = self.metrics['avg_reward'][-window:]
        recent_violations = self.metrics['constraint_violations'][-window:]
        
        summary = {
            'success_rate': np.mean(recent_success),
            'cost_guarantee_rate': np.mean(self.metrics['cost_guarantee_rate'][-window:]),
            'avg_cost': np.mean(recent_cost),
            'avg_reward': np.mean(recent_reward),
            'constraint_violation_rate': np.mean([v > 0 for v in recent_violations]),
            'avg_episode_length': np.mean(self.metrics['episode_lengths'][-window:]),
            'total_episodes': len(self.metrics['success_rate'])
        }
        
        # Add method-specific metrics
        if self.method == 'ppo_lagrangian' and self.metrics['duality_gap']:
            summary['avg_duality_gap'] = np.mean(self.metrics['duality_gap'][-window:])
        
        # Add LLM orchestrator specific metrics
        if self.use_llm_orchestrator and 'replanning_count' in self.metrics:
            summary['avg_replanning_count'] = np.mean(self.metrics['replanning_count'][-window:])
            summary['replanning_rate'] = np.mean([r > 0 for r in self.metrics['replanning_count'][-window:]])
        
        return summary
    
    def get_enhanced_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics including WebArenaMetrics if available"""
        summary = {
            'basic_metrics': self.get_metrics_summary(),
            'method': self.method,
            'use_llm_orchestrator': self.use_llm_orchestrator,
            'total_episodes': self.episode_count
        }
        
        if self.webarena_metrics:
            summary['enhanced_metrics'] = {
                'task_summary': self.webarena_metrics.get_task_summary(),
                'llm_orchestrator_metrics': self.webarena_metrics.get_llm_orchestrator_metrics(),
                'cost_analysis': self.webarena_metrics.get_cost_analysis(),
                'efficiency_metrics': self.webarena_metrics.get_efficiency_metrics()
            }
        
        return summary
    
    def save_checkpoint(self, filepath: str):
        """Save complete system checkpoint"""
        checkpoint = {
            'method': self.method,
            'episode_count': self.episode_count,
            'metrics': self.metrics,
            'algorithm_state': None,
            'use_llm_orchestrator': self.use_llm_orchestrator
        }
        
        # Save algorithm state
        if hasattr(self.algorithm, 'save_checkpoint'):
            algorithm_path = filepath.replace('.pt', '_algorithm.pt')
            self.algorithm.save_checkpoint(algorithm_path)
            checkpoint['algorithm_path'] = algorithm_path
        
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
        
        print(f"Checkpoint loaded from {filepath}")


# Utility functions for integration with existing WebArena
def create_enhanced_mas_from_config(config_path: str) -> EnhancedWebArenaMAS:
    """Create EnhancedWebArenaMAS from configuration file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return EnhancedWebArenaMAS(**config)


def test_enhanced_metrics_integration():
    """Test the enhanced metrics integration"""
    print("Testing Enhanced WebArena MAS with Metrics Integration...")
    
    # Create MAS with LLM orchestrator
    mas = EnhancedWebArenaMAS(
        method='p3o',
        budget=1.0,
        use_llm_orchestrator=True,
        llm_model='gpt-4-turbo',
        enable_replanning=True,
        num_agents=2
    )
    
    # Test task
    test_task = {
        'intent': 'Test task with metrics',
        'sites': ['example.com'],
        'expected_steps': 3,
        'observation': 'Test homepage'
    }
    
    try:
        result = mas.solve_task(test_task)
        print(f"✅ Task completed - Success: {result['success']}, Cost: {result['cost']:.3f}")
        
        if 'enhanced_metrics' in result:
            print("✅ Enhanced metrics captured successfully")
            metrics = result['enhanced_metrics']
            if 'task_summary' in metrics:
                print(f"   - Task summary available")
            if 'llm_orchestrator_metrics' in metrics:
                print(f"   - LLM orchestrator metrics available")
            if 'cost_analysis' in metrics:
                print(f"   - Cost analysis available")
        else:
            print("⚠️  Enhanced metrics not available (WebArenaMetrics not imported)")
        
        # Test metrics summary
        summary = mas.get_enhanced_metrics_summary()
        print(f"✅ Enhanced metrics summary: {len(summary)} sections")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def run_single_task_test(method: str = 'p3o', use_llm: bool = False) -> Dict:
    """Run a single task test for debugging"""
    # Create test task
    test_task = {
        'task_id': 'test_001',
        'intent': 'Navigate to shopping page and add item to cart',
        'sites': ['shopping.com'],
        'expected_steps': 3,
        'observation': 'Current page shows shopping website homepage with search bar visible'
    }
    
    # Create MAS
    mas = EnhancedWebArenaMAS(
        method=method, 
        budget=0.5, 
        num_agents=2,
        use_llm_orchestrator=use_llm,
        llm_model="gpt-4-turbo" if use_llm else None
    )
    
    # Solve task
    result = mas.solve_task(test_task)
    
    print(f"Test Result - Method: {method}, LLM: {use_llm}")
    print(f"Success: {result['success']}")
    print(f"Cost: {result['cost']:.3f} (Budget: {mas.budget})")
    print(f"Reward: {result['reward']:.3f}")
    print(f"DAG nodes: {result['dag_metrics'].get('nodes', 0)}")
    print(f"Replanning count: {result.get('replanning_count', 0)}")
    
    # Enhanced metrics information
    if 'enhanced_metrics' in result:
        print("✅ Enhanced metrics collected successfully")
    
    if use_llm and 'dag' in result and isinstance(result['dag'], dict):
        dag_info = result['dag'].get('dag', {})
        print(f"LLM confidence: {dag_info.get('confidence', 'N/A')}")
        print(f"LLM reasoning: {dag_info.get('reasoning', 'N/A')}")
    
    return result


if __name__ == "__main__":
    # Run tests
    print("Enhanced WebArena MAS with Metrics Integration")
    print("=" * 50)
    
    # Test basic functionality
    print("=== Testing Regular Orchestrator ===")
    result1 = run_single_task_test('p3o', use_llm=False)
    
    print("\n=== Testing LLM Orchestrator ===")
    result2 = run_single_task_test('p3o', use_llm=True)
    
    print("\n=== Testing Enhanced Metrics Integration ===")
    test_enhanced_metrics_integration()