import networkx as nx
from typing import Dict, List, Tuple, Any, Optional

class DAGValidator:
    """Validates DAG structures and provides fallback generation"""
    
    def __init__(self, max_nodes: int = 10, num_agents: int = 4):
        self.max_nodes = max_nodes
        self.num_agents = num_agents
        
    def validate_complete(self, dag_json: Dict, context: Dict) -> Tuple[bool, List[str]]:
        """
        Complete validation of DAG structure and constraints
        
        Args:
            dag_json: Generated DAG in JSON format
            context: Task context for additional validation
            
        Returns:
            is_valid: Boolean indicating if DAG is valid
            errors: List of validation error messages
        """
        errors = []
        
        # Check basic structure
        structure_valid, structure_errors = self.validate_structure(dag_json)
        errors.extend(structure_errors)
        
        # Check DAG properties
        dag_valid, dag_errors = self.validate_dag_properties(dag_json)
        errors.extend(dag_errors)
        
        # Check agent assignments
        agent_valid, agent_errors = self.validate_agent_assignments(dag_json)
        errors.extend(agent_errors)
        
        # Check cost constraints
        cost_valid, cost_errors = self.validate_cost_constraints(dag_json, context)
        errors.extend(cost_errors)
        
        # Check parallel groups
        parallel_valid, parallel_errors = self.validate_parallel_groups(dag_json)
        errors.extend(parallel_errors)
        
        is_valid = all([structure_valid, dag_valid, agent_valid, cost_valid, parallel_valid])
        
        return is_valid, errors
    
    def validate_structure(self, dag_json: Dict) -> Tuple[bool, List[str]]:
        """Validate basic DAG structure"""
        errors = []
        
        try:
            # Check top-level structure
            if 'dag' not in dag_json:
                errors.append("Missing 'dag' key in root")
                return False, errors
            
            dag = dag_json['dag']
            required_keys = ['nodes', 'edges', 'agent_assignments']
            for key in required_keys:
                if key not in dag:
                    errors.append(f"Missing required key: {key}")
            
            # Validate nodes
            if 'nodes' in dag:
                if not isinstance(dag['nodes'], list):
                    errors.append("'nodes' must be a list")
                else:
                    for i, node in enumerate(dag['nodes']):
                        if not isinstance(node, dict):
                            errors.append(f"Node {i} must be a dictionary")
                            continue
                        
                        required_node_keys = ['id', 'task', 'estimated_cost']
                        for key in required_node_keys:
                            if key not in node:
                                errors.append(f"Node {i} missing required key: {key}")
                        
                        # Check node ID is integer
                        if 'id' in node and not isinstance(node['id'], int):
                            errors.append(f"Node {i} ID must be an integer")
                        
                        # Check cost is numeric
                        if 'estimated_cost' in node and not isinstance(node['estimated_cost'], (int, float)):
                            errors.append(f"Node {i} estimated_cost must be numeric")
            
            # Validate edges
            if 'edges' in dag:
                if not isinstance(dag['edges'], list):
                    errors.append("'edges' must be a list")
                else:
                    for i, edge in enumerate(dag['edges']):
                        if not isinstance(edge, list) or len(edge) != 2:
                            errors.append(f"Edge {i} must be a list of length 2")
                        elif not all(isinstance(x, int) for x in edge):
                            errors.append(f"Edge {i} must contain integers")
            
            # Validate agent assignments
            if 'agent_assignments' in dag:
                if not isinstance(dag['agent_assignments'], dict):
                    errors.append("'agent_assignments' must be a dictionary")
        
        except Exception as e:
            errors.append(f"Structure validation exception: {str(e)}")
        
        return len(errors) == 0, errors
    
    def validate_dag_properties(self, dag_json: Dict) -> Tuple[bool, List[str]]:
        """Validate that the graph is a valid DAG"""
        errors = []
        
        try:
            dag = dag_json['dag']
            nodes = dag.get('nodes', [])
            edges = dag.get('edges', [])
            
            if not nodes:
                errors.append("DAG must have at least one node")
                return False, errors
            
            # Check node count
            if len(nodes) > self.max_nodes:
                errors.append(f"Too many nodes: {len(nodes)} > {self.max_nodes}")
            
            # Create NetworkX graph to check DAG property
            G = nx.DiGraph()
            
            # Add nodes
            for node in nodes:
                if 'id' in node:
                    G.add_node(node['id'])
            
            # Add edges
            node_ids = set(node['id'] for node in nodes if 'id' in node)
            for edge in edges:
                if len(edge) == 2:
                    src, dst = edge
                    if src not in node_ids:
                        errors.append(f"Edge source {src} not found in nodes")
                    if dst not in node_ids:
                        errors.append(f"Edge destination {dst} not found in nodes")
                    if src in node_ids and dst in node_ids:
                        G.add_edge(src, dst)
            
            # Check if it's a DAG
            if not nx.is_directed_acyclic_graph(G):
                errors.append("Graph contains cycles - not a valid DAG")
                
                # Try to identify cycles
                try:
                    cycles = list(nx.simple_cycles(G))
                    if cycles:
                        errors.append(f"Found cycles: {cycles[:3]}")  # Show first 3 cycles
                except:
                    pass
            
            # Check connectivity (should have a path from start to end nodes)
            if len(G.nodes()) > 1:
                # Find nodes with no incoming edges (start nodes)
                start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
                # Find nodes with no outgoing edges (end nodes)
                end_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
                
                if not start_nodes:
                    errors.append("No start nodes found (nodes with no incoming edges)")
                if not end_nodes:
                    errors.append("No end nodes found (nodes with no outgoing edges)")
        
        except Exception as e:
            errors.append(f"DAG validation exception: {str(e)}")
        
        return len(errors) == 0, errors
    
    def validate_agent_assignments(self, dag_json: Dict) -> Tuple[bool, List[str]]:
        """Validate agent assignments"""
        errors = []
        
        try:
            dag = dag_json['dag']
            nodes = dag.get('nodes', [])
            assignments = dag.get('agent_assignments', {})
            
            node_ids = set(str(node['id']) for node in nodes if 'id' in node)
            
            # Check that all nodes have agent assignments
            for node in nodes:
                if 'id' in node:
                    node_id_str = str(node['id'])
                    if node_id_str not in assignments:
                        errors.append(f"Node {node['id']} has no agent assignment")
            
            # Check that assignments reference valid nodes
            for node_id_str, agent in assignments.items():
                if node_id_str not in node_ids:
                    errors.append(f"Agent assignment for non-existent node: {node_id_str}")
                
                # Validate agent name (basic check)
                if not isinstance(agent, str) or not agent.strip():
                    errors.append(f"Invalid agent name for node {node_id_str}: {agent}")
        
        except Exception as e:
            errors.append(f"Agent assignment validation exception: {str(e)}")
        
        return len(errors) == 0, errors
    
    def validate_cost_constraints(self, dag_json: Dict, context: Dict) -> Tuple[bool, List[str]]:
        """Validate cost constraints"""
        errors = []
        
        try:
            dag = dag_json['dag']
            nodes = dag.get('nodes', [])
            
            # Calculate total estimated cost
            total_cost = 0
            for node in nodes:
                cost = node.get('estimated_cost', 0)
                if isinstance(cost, (int, float)):
                    total_cost += cost
                else:
                    errors.append(f"Invalid cost for node {node.get('id', 'unknown')}: {cost}")
            
            # Check against remaining budget
            constraints = context.get('constraints', {})
            budget = constraints.get('budget', 1.0)
            cost_spent = constraints.get('cost_spent', 0)
            remaining_budget = budget - cost_spent
            
            if total_cost > remaining_budget:
                errors.append(f"Total estimated cost {total_cost:.3f} exceeds remaining budget {remaining_budget:.3f}")
            
            # Check individual node costs are reasonable
            for node in nodes:
                cost = node.get('estimated_cost', 0)
                if cost > remaining_budget * 0.5:  # No single node should use more than 50% of remaining budget
                    errors.append(f"Node {node.get('id', 'unknown')} cost {cost:.3f} too high (>{remaining_budget*0.5:.3f})")
                if cost <= 0:
                    errors.append(f"Node {node.get('id', 'unknown')} has non-positive cost: {cost}")
        
        except Exception as e:
            errors.append(f"Cost validation exception: {str(e)}")
        
        return len(errors) == 0, errors
    
    def validate_parallel_groups(self, dag_json: Dict) -> Tuple[bool, List[str]]:
        """Validate parallel execution groups"""
        errors = []
        
        try:
            dag = dag_json['dag']
            nodes = dag.get('nodes', [])
            edges = dag.get('edges', [])
            parallel_groups = dag.get('parallel_groups', [])
            
            if not parallel_groups:
                # Not required, but warn if missing
                return True, []
            
            node_ids = set(node['id'] for node in nodes if 'id' in node)
            
            # Build dependency graph
            dependencies = {node_id: set() for node_id in node_ids}
            for edge in edges:
                if len(edge) == 2:
                    src, dst = edge
                    if dst in dependencies:
                        dependencies[dst].add(src)
            
            # Check each parallel group
            for i, group in enumerate(parallel_groups):
                if not isinstance(group, list):
                    errors.append(f"Parallel group {i} must be a list")
                    continue
                
                # Check that nodes in the group can actually run in parallel
                for j, node_id in enumerate(group):
                    if node_id not in node_ids:
                        errors.append(f"Parallel group {i} contains non-existent node: {node_id}")
                        continue
                    
                    # Check that nodes in this group don't depend on each other
                    for k, other_node_id in enumerate(group):
                        if j != k and other_node_id in dependencies.get(node_id, set()):
                            errors.append(f"Parallel group {i}: node {node_id} depends on {other_node_id} in same group")
        
        except Exception as e:
            errors.append(f"Parallel groups validation exception: {str(e)}")
        
        return len(errors) == 0, errors
    
    def generate_fallback_dag(self, context: Dict) -> Dict:
        """Generate a simple fallback DAG when validation fails"""
        
        # Extract basic info from context
        task = context.get('task', {})
        task_type = task.get('intent', 'unknown_task')
        expected_steps = min(task.get('expected_steps', 3), self.max_nodes)
        
        # Get budget constraints
        constraints = context.get('constraints', {})
        budget = constraints.get('budget', 1.0)
        cost_spent = constraints.get('cost_spent', 0)
        remaining_budget = budget - cost_spent
        
        # Generate simple sequential nodes
        nodes = []
        edges = []
        agent_assignments = {}
        
        cost_per_step = min(remaining_budget / max(expected_steps, 1), 0.1)
        
        for i in range(expected_steps):
            nodes.append({
                'id': i,
                'task': f'step_{i+1}',
                'description': f'Step {i+1} for {task_type}',
                'estimated_cost': cost_per_step,
                'difficulty': 0.5,
                'priority': 'medium',
                'expected_reward': 0.1
            })
            
            # Sequential edges
            if i > 0:
                edges.append([i-1, i])
            
            # Assign to a default agent
            agent_assignments[str(i)] = 'gpt-4-turbo'
        
        # Create parallel groups (sequential for safety)
        parallel_groups = [[i] for i in range(expected_steps)]
        
        return {
            'dag': {
                'nodes': nodes,
                'edges': edges,
                'agent_assignments': agent_assignments,
                'parallel_groups': parallel_groups,
                'checkpoint_nodes': [expected_steps // 2] if expected_steps > 2 else [],
                'confidence': 0.5,
                'estimated_total_cost': cost_per_step * expected_steps,
                'reasoning': 'Fallback DAG generated due to validation failure'
            }
        }