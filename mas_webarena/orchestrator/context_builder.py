from typing import Dict, List, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.observation_processor import ObservationProcessor

def build_context(
    task: Dict,
    current_observation: str,
    trajectory: List[Dict],
    budget_tracker: 'BudgetTracker',
    episode: int = 0,
    method: str = "p3o"
) -> Dict:
    """Build complete context for LLM orchestrator"""
    
    # Parse observation to get page state
    obs_processor = ObservationProcessor()
    processed_obs = obs_processor.process(current_observation)
    
    context = {
        "task": task,
        "constraints": {
            "budget": budget_tracker.initial_budget,
            "alpha": 1.05,
            "beta": 0.95,
            "cost_spent": budget_tracker.spent,
            "current_method": method
        },
        "agent_pool": {
            "claude-3.5": {
                "tags": ["reasoning", "complex_ui", "high_cost"],
                "availability": True,
                "cost_per_action": 0.15,
                "capabilities": ["click", "type", "navigate", "verify"],
                "success_rate": 0.85
            },
            "gpt-4-turbo": {
                "tags": ["fast", "reliable", "medium_cost"],
                "availability": True,
                "cost_per_action": 0.10,
                "capabilities": ["click", "type", "navigate", "verify"],
                "success_rate": 0.80
            },
            "kimi-k2": {
                "tags": ["moe", "efficient", "low_cost"],
                "availability": True,
                "cost_per_action": 0.05,
                "capabilities": ["click", "type", "navigate"],
                "success_rate": 0.70
            },
            "gemini-1.5": {
                "tags": ["multimodal", "fast", "medium_cost"],
                "availability": True,
                "cost_per_action": 0.08,
                "capabilities": ["click", "type", "navigate", "verify"],
                "success_rate": 0.75
            }
        },
        "current_state": {
            "observation": current_observation,
            "page_type": processed_obs.get('type', 'unknown'),
            "elements_available": [e.get('tag', 'unknown') for e in processed_obs.get('elements', [])],
            "previous_actions": [t.get('action', 'unknown') for t in trajectory[-3:]]  # Last 3 actions
        },
        "execution_history": {
            "successful_steps": [i for i, t in enumerate(trajectory) if t.get('success', False)],
            "failed_attempts": [t for t in trajectory if not t.get('success', False)],
            "checkpoint_states": {},  # Would be populated by checkpoint system
            "trajectory": trajectory,
            "replanning_count": sum(1 for t in trajectory if t.get('replanned', False))
        },
        "learning_info": {
            "episode": episode,
            "duality_gap": 0.1,  # Would come from PPO-Lagrangian
            "penalty_coef": 10.0,  # Would come from P3O
            "lambda_coord": 0.5   # Would come from MACPO
        }
    }
    
    return context

def build_simplified_context(
    task_intent: str,
    page_type: str = "unknown",
    elements: List[str] = None,
    budget_remaining: float = 1.0,
    method: str = "p3o"
) -> Dict:
    """Build a simplified context for testing or simple scenarios"""
    
    if elements is None:
        elements = []
    
    return {
        "task": {
            "intent": task_intent,
            "sites": ["example.com"],
            "difficulty": "medium",
            "expected_steps": 5
        },
        "constraints": {
            "budget": budget_remaining + 0.1,  # Add small buffer
            "alpha": 1.05,
            "beta": 0.95,
            "cost_spent": 0.1,  # Assume some cost already spent
            "current_method": method
        },
        "agent_pool": {
            "gpt-4-turbo": {
                "tags": ["fast", "reliable", "medium_cost"],
                "availability": True,
                "cost_per_action": 0.10,
                "capabilities": ["click", "type", "navigate", "verify"],
                "success_rate": 0.80
            },
            "kimi-k2": {
                "tags": ["moe", "efficient", "low_cost"],
                "availability": True,
                "cost_per_action": 0.05,
                "capabilities": ["click", "type", "navigate"],
                "success_rate": 0.70
            }
        },
        "current_state": {
            "observation": f"Current page: {page_type}",
            "page_type": page_type,
            "elements_available": elements,
            "previous_actions": []
        },
        "execution_history": {
            "successful_steps": [],
            "failed_attempts": [],
            "checkpoint_states": {},
            "trajectory": [],
            "replanning_count": 0
        },
        "learning_info": {
            "episode": 1,
            "duality_gap": 0.1,
            "penalty_coef": 10.0,
            "lambda_coord": 0.5
        }
    }

def update_context_with_result(context: Dict, result: Dict) -> Dict:
    """Update context with execution result"""
    updated_context = context.copy()
    
    # Update cost spent
    if 'cost' in result:
        updated_context['constraints']['cost_spent'] += result['cost']
    
    # Update current state based on result
    if 'observation' in result:
        # Parse new observation
        obs_processor = ObservationProcessor()
        processed_obs = obs_processor.process(result['observation'])
        
        updated_context['current_state']['observation'] = result['observation']
        updated_context['current_state']['page_type'] = processed_obs.get('type', 'unknown')
        updated_context['current_state']['elements_available'] = [
            e.get('tag', 'unknown') for e in processed_obs.get('elements', [])
        ]
    
    # Update trajectory
    trajectory_entry = {
        'action': result.get('action', 'unknown'),
        'success': result.get('success', False),
        'cost': result.get('cost', 0),
        'reward': result.get('reward', 0),
        'timestamp': result.get('timestamp', 0)
    }
    
    updated_context['execution_history']['trajectory'].append(trajectory_entry)
    
    # Update previous actions
    updated_context['current_state']['previous_actions'].append(result.get('action', 'unknown'))
    # Keep only last 3 actions
    updated_context['current_state']['previous_actions'] = \
        updated_context['current_state']['previous_actions'][-3:]
    
    # Update successful/failed steps
    if result.get('success', False):
        step_index = len(updated_context['execution_history']['trajectory']) - 1
        updated_context['execution_history']['successful_steps'].append(step_index)
    else:
        updated_context['execution_history']['failed_attempts'].append(trajectory_entry)
    
    return updated_context

def extract_task_features(context: Dict) -> Dict:
    """Extract key features from context for analysis"""
    task = context.get('task', {})
    current_state = context.get('current_state', {})
    constraints = context.get('constraints', {})
    
    features = {
        # Task characteristics
        'intent': task.get('intent', ''),
        'sites': task.get('sites', []),
        'difficulty': task.get('difficulty', 'medium'),
        'expected_steps': task.get('expected_steps', 5),
        
        # State characteristics
        'page_type': current_state.get('page_type', 'unknown'),
        'num_elements': len(current_state.get('elements_available', [])),
        'has_search_elements': any('search' in elem.lower() 
                                 for elem in current_state.get('elements_available', [])),
        'has_form_elements': any(elem.lower() in ['input', 'button', 'select'] 
                               for elem in current_state.get('elements_available', [])),
        
        # Budget characteristics
        'budget_remaining': constraints.get('budget', 1.0) - constraints.get('cost_spent', 0),
        'budget_utilization': constraints.get('cost_spent', 0) / max(constraints.get('budget', 1.0), 0.01),
        'method': constraints.get('current_method', 'p3o'),
        
        # History characteristics
        'trajectory_length': len(context.get('execution_history', {}).get('trajectory', [])),
        'success_rate': len(context.get('execution_history', {}).get('successful_steps', [])) / 
                       max(len(context.get('execution_history', {}).get('trajectory', [])), 1),
        'replanning_count': context.get('execution_history', {}).get('replanning_count', 0)
    }
    
    return features

def validate_context(context: Dict) -> List[str]:
    """Validate that context has all required fields"""
    errors = []
    
    required_top_level = ['task', 'constraints', 'agent_pool', 'current_state']
    for key in required_top_level:
        if key not in context:
            errors.append(f"Missing required top-level key: {key}")
    
    # Validate task
    if 'task' in context:
        task = context['task']
        required_task_keys = ['intent']
        for key in required_task_keys:
            if key not in task:
                errors.append(f"Missing required task key: {key}")
    
    # Validate constraints
    if 'constraints' in context:
        constraints = context['constraints']
        required_constraint_keys = ['budget', 'cost_spent']
        for key in required_constraint_keys:
            if key not in constraints:
                errors.append(f"Missing required constraint key: {key}")
            elif not isinstance(constraints[key], (int, float)):
                errors.append(f"Constraint {key} must be numeric")
    
    # Validate agent pool
    if 'agent_pool' in context:
        agent_pool = context['agent_pool']
        if not isinstance(agent_pool, dict) or not agent_pool:
            errors.append("Agent pool must be a non-empty dictionary")
        else:
            for agent_name, agent_info in agent_pool.items():
                if not isinstance(agent_info, dict):
                    errors.append(f"Agent {agent_name} info must be a dictionary")
                elif 'cost_per_action' not in agent_info:
                    errors.append(f"Agent {agent_name} missing cost_per_action")
    
    # Validate current state
    if 'current_state' in context:
        current_state = context['current_state']
        if 'page_type' not in current_state:
            errors.append("Missing page_type in current_state")
    
    return errors