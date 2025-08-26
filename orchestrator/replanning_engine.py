import time
from typing import Dict, List, Tuple, Optional, Any

class ReplanningEngine:
    """Handles dynamic re-planning decisions and execution"""
    
    def __init__(
        self, 
        orchestrator: 'LLMOrchestratorPolicy',
        replan_threshold: float = 0.3,
        max_replans_per_task: int = 3
    ):
        self.orchestrator = orchestrator
        self.replan_threshold = replan_threshold
        self.max_replans_per_task = max_replans_per_task
        self.replan_history = []
        
    def should_replan(
        self, 
        current_state: Dict,
        trajectory: List[Dict],
        current_dag: Dict
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if re-planning is needed
        
        Returns:
            should_replan: Boolean indicating if re-planning needed
            reason: String describing why re-planning is needed
        """
        # Check if we've exceeded max replans
        replans_so_far = sum(1 for t in trajectory if t.get('replanned', False))
        if replans_so_far >= self.max_replans_per_task:
            return False, None
            
        reasons = []
        
        # 1. Direct failure detection
        if current_state.get('last_action_failed', False):
            reasons.append('action_failure')
            
        # 2. Budget inefficiency check
        cost_spent = current_state.get('constraints', {}).get('cost_spent', 0)
        budget = current_state.get('constraints', {}).get('budget', 1.0)
        expected_steps = current_state.get('task', {}).get('expected_steps', 1)
        progress = len(trajectory) / max(1, expected_steps)
        
        if cost_spent > 0.5 * budget and progress < self.replan_threshold:
            reasons.append(f'budget_inefficiency (spent {cost_spent:.2f}, progress {progress:.2%})')
            
        # 3. Constraint violation
        alpha = current_state.get('constraints', {}).get('alpha', 1.05)
        if cost_spent > budget * alpha:
            reasons.append(f'constraint_violation (cost {cost_spent:.2f} > {budget * alpha:.2f})')
            
        # 4. Unexpected state detection
        if self._detect_unexpected_state(current_state, current_dag):
            reasons.append('unexpected_state')
            
        # 5. Low confidence check
        confidence = current_dag.get('dag', {}).get('confidence', 1.0)
        if confidence < 0.5:
            reasons.append(f'low_confidence ({confidence:.2f})')
            
        # 6. Repeated failures
        recent_failures = sum(1 for t in trajectory[-3:] if not t.get('success', True))
        if recent_failures >= 2:
            reasons.append(f'repeated_failures ({recent_failures} in last 3 steps)')
            
        # 7. Stuck detection (no progress)
        if len(trajectory) > 5:
            recent_rewards = [t.get('reward', 0) for t in trajectory[-5:]]
            if all(r <= 0 for r in recent_rewards):
                reasons.append('no_progress (no positive rewards in last 5 steps)')
        
        should_replan = len(reasons) > 0
        reason = ' AND '.join(reasons) if should_replan else None
        
        return should_replan, reason
    
    def execute_replanning(
        self,
        current_dag: Dict,
        current_state: Dict,
        reason: str,
        completed_nodes: List[int]
    ) -> Dict:
        """
        Execute re-planning and return new DAG
        
        Args:
            current_dag: Current DAG that needs replanning
            current_state: Current execution state
            reason: Reason for replanning
            completed_nodes: List of already completed node IDs
            
        Returns:
            new_dag: Updated DAG
        """
        # Find the failure point
        all_nodes = [n['id'] for n in current_dag['dag']['nodes']]
        remaining_nodes = [n for n in all_nodes if n not in completed_nodes]
        
        if not remaining_nodes:
            # All nodes completed, no need to replan
            return current_dag
            
        failure_point = remaining_nodes[0] if remaining_nodes else None
        
        # Prepare replanning context
        replan_context = {
            **current_state,
            'replanning': {
                'original_dag': current_dag,
                'completed_nodes': completed_nodes,
                'remaining_nodes': remaining_nodes,
                'failure_point': failure_point,
                'reason': reason,
                'checkpoint_state': self._get_latest_checkpoint(current_state, completed_nodes)
            }
        }
        
        # Generate new plan using LLM
        try:
            # Use prompt manager to format replanning prompt
            from .prompt_manager import PromptManager
            prompt_manager = PromptManager()
            
            remaining_budget = (current_state.get('constraints', {}).get('budget', 1.0) - 
                              current_state.get('constraints', {}).get('cost_spent', 0))
            
            prompt = prompt_manager.format_replanning_prompt(
                context=current_state,
                reason=reason,
                failure_point=failure_point,
                completed_nodes=completed_nodes,
                original_dag=current_dag,
                remaining_budget=remaining_budget
            )
            
            # Generate new plan (this would use the LLM client)
            new_dag, _ = self.orchestrator.generate_dag(replan_context, use_cache=False)
            
        except Exception as e:
            print(f"Replanning generation failed: {e}")
            # Fall back to simple modification of existing plan
            new_dag = self._generate_simple_replan(current_dag, current_state, completed_nodes)
        
        # Merge the new plan with completed portions
        merged_dag = self._merge_dags(current_dag, new_dag, completed_nodes)
        
        # Record replanning event
        self.replan_history.append({
            'timestamp': time.time(),
            'reason': reason,
            'original_dag': current_dag,
            'new_dag': merged_dag,
            'failure_point': failure_point,
            'completed_nodes': completed_nodes
        })
        
        return merged_dag
    
    def _detect_unexpected_state(self, current_state: Dict, current_dag: Dict) -> bool:
        """Detect if current state differs from expected"""
        expected_elements = current_dag.get('expected_state', {}).get('elements', [])
        actual_elements = current_state.get('current_state', {}).get('elements_available', [])
        
        # Check if key expected elements are missing
        if expected_elements and actual_elements:
            missing = set(expected_elements) - set(actual_elements)
            return len(missing) > len(expected_elements) * 0.5
        
        # Check if page type changed unexpectedly
        expected_page = current_dag.get('expected_state', {}).get('page_type', '')
        actual_page = current_state.get('current_state', {}).get('page_type', '')
        if expected_page and actual_page and expected_page != actual_page:
            return True
            
        return False
    
    def _get_latest_checkpoint(self, state: Dict, completed_nodes: List[int]) -> Optional[Dict]:
        """Get the most recent checkpoint state"""
        checkpoints = state.get('execution_history', {}).get('checkpoint_states', {})
        # Find latest checkpoint from completed nodes
        for node_id in reversed(completed_nodes):
            if str(node_id) in checkpoints:
                return checkpoints[str(node_id)]
        return None
    
    def _generate_simple_replan(self, original_dag: Dict, current_state: Dict, completed_nodes: List[int]) -> Dict:
        """Generate a simple replan by modifying the existing DAG"""
        # Create a simplified version with remaining nodes
        original_nodes = original_dag['dag']['nodes']
        remaining_nodes = [n for n in original_nodes if n['id'] not in completed_nodes]
        
        # Adjust node costs and add verification steps
        new_nodes = []
        for i, node in enumerate(remaining_nodes):
            # Create new node with adjusted parameters
            new_node = node.copy()
            new_node['id'] = i
            
            # Reduce cost estimate (be more conservative)
            new_node['estimated_cost'] = min(node.get('estimated_cost', 0.05) * 0.8, 0.05)
            
            # Add verification if this was a failure point
            if node['id'] == remaining_nodes[0]['id']:  # First remaining node likely failed
                new_node['task'] = f"retry_{node['task']}"
                new_node['description'] = f"Retry with verification: {node.get('description', '')}"
                
            new_nodes.append(new_node)
        
        # Create simple sequential edges
        new_edges = []
        for i in range(len(new_nodes) - 1):
            new_edges.append([i, i + 1])
        
        # Use more reliable agents
        new_assignments = {}
        for node in new_nodes:
            # Prefer more reliable but potentially more expensive agents
            new_assignments[str(node['id'])] = 'claude-3.5'  # Most reliable
        
        return {
            'dag': {
                'nodes': new_nodes,
                'edges': new_edges,
                'agent_assignments': new_assignments,
                'parallel_groups': [[i] for i in range(len(new_nodes))],  # Sequential execution
                'checkpoint_nodes': [len(new_nodes) // 2] if len(new_nodes) > 2 else [],
                'confidence': 0.6,
                'estimated_total_cost': sum(n['estimated_cost'] for n in new_nodes),
                'reasoning': 'Simple replan with conservative estimates and reliable agents'
            }
        }
    
    def _merge_dags(self, original_dag: Dict, new_dag: Dict, completed_nodes: List[int]) -> Dict:
        """Merge new DAG with completed portions of original"""
        merged = {
            'dag': {
                'nodes': [],
                'edges': [],
                'agent_assignments': {},
                'parallel_groups': [],
                'checkpoint_nodes': original_dag['dag'].get('checkpoint_nodes', []),
                'confidence': new_dag['dag'].get('confidence', 0.5)
            }
        }
        
        # Keep completed nodes from original
        for node in original_dag['dag']['nodes']:
            if node['id'] in completed_nodes:
                merged['dag']['nodes'].append(node)
                # Keep agent assignment
                node_id_str = str(node['id'])
                if node_id_str in original_dag['dag']['agent_assignments']:
                    merged['dag']['agent_assignments'][node_id_str] = \
                        original_dag['dag']['agent_assignments'][node_id_str]
        
        # Add new nodes with updated IDs
        id_offset = max(completed_nodes) + 1 if completed_nodes else 0
        for node in new_dag['dag']['nodes']:
            new_node = node.copy()
            new_node['id'] = node['id'] + id_offset
            merged['dag']['nodes'].append(new_node)
            # Update agent assignments
            old_id_str = str(node['id'])
            new_id_str = str(new_node['id'])
            if old_id_str in new_dag['dag']['agent_assignments']:
                merged['dag']['agent_assignments'][new_id_str] = \
                    new_dag['dag']['agent_assignments'][old_id_str]
        
        # Rebuild edges
        merged['dag']['edges'] = self._rebuild_edges(merged['dag']['nodes'], completed_nodes)
        
        # Update parallel groups
        merged['dag']['parallel_groups'] = self._rebuild_parallel_groups(
            merged['dag']['nodes'],
            merged['dag']['edges'],
            completed_nodes
        )
        
        # Update other fields
        merged['dag']['estimated_total_cost'] = sum(n.get('estimated_cost', 0) for n in merged['dag']['nodes'])
        merged['dag']['reasoning'] = f"Merged replan: {new_dag['dag'].get('reasoning', 'Unknown')}"
        
        return merged
    
    def _rebuild_edges(self, nodes: List[Dict], completed_nodes: List[int]) -> List[List[int]]:
        """Rebuild edges based on node dependencies"""
        edges = []
        node_ids = [n['id'] for n in nodes]
        completed_set = set(completed_nodes)
        
        # Connect completed nodes to first new node
        if completed_nodes and len(node_ids) > len(completed_nodes):
            last_completed = max(completed_nodes)
            first_new = min(n['id'] for n in nodes if n['id'] not in completed_set)
            edges.append([last_completed, first_new])
        
        # Create sequential connections for new nodes
        new_nodes = [n for n in nodes if n['id'] not in completed_set]
        for i, node in enumerate(new_nodes[:-1]):
            edges.append([node['id'], new_nodes[i+1]['id']])
            
        return edges
    
    def _rebuild_parallel_groups(
        self, 
        nodes: List[Dict], 
        edges: List[List[int]], 
        completed_nodes: List[int]
    ) -> List[List[int]]:
        """Identify which nodes can execute in parallel"""
        # Build dependency graph
        dependencies = {n['id']: set() for n in nodes}
        for edge in edges:
            if len(edge) == 2:
                dependencies[edge[1]].add(edge[0])
        
        # Group nodes by dependency level
        levels = []
        remaining = set(n['id'] for n in nodes if n['id'] not in completed_nodes)
        
        # Completed nodes are in their own "level" (already done)
        if completed_nodes:
            levels.append(completed_nodes)
        
        while remaining:
            # Find nodes with no dependencies in remaining set
            current_level = []
            for node_id in remaining:
                deps_in_remaining = dependencies[node_id] & remaining
                if not deps_in_remaining:
                    current_level.append(node_id)
            
            if not current_level:
                # Circular dependency, break by taking arbitrary node
                current_level = [remaining.pop()]
                
            levels.append(current_level)
            remaining -= set(current_level)
            
        return levels
    
    def get_replan_stats(self) -> Dict:
        """Get statistics about replanning events"""
        if not self.replan_history:
            return {
                'total_replans': 0,
                'avg_time_between_replans': 0,
                'most_common_reasons': [],
                'success_after_replan': 0
            }
        
        # Count reasons
        reason_counts = {}
        for event in self.replan_history:
            reason = event['reason']
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # Sort by frequency
        most_common = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate average time between replans
        timestamps = [event['timestamp'] for event in self.replan_history]
        if len(timestamps) > 1:
            time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            avg_time = sum(time_diffs) / len(time_diffs)
        else:
            avg_time = 0
        
        return {
            'total_replans': len(self.replan_history),
            'avg_time_between_replans': avg_time,
            'most_common_reasons': most_common[:5],
            'replan_events': len(self.replan_history)
        }