# webarena_wrapper.py
import time
from typing import Dict, List, Any
from browser_env import ScriptBrowserEnv
from evaluation_harness.evaluators import Evaluator
from webArenaCostMonitor import WebArenaCostMonitor
from CommunicationMonitor import CommunicationMonitor
from ActionMonitor import ActionMonitor

class WebArenaWrapper:
    """Wrapper for WebArena that adds cost and communication monitoring"""
    
    def __init__(self, env_config=None):
        # Initialize WebArena components
        self.env = ScriptBrowserEnv(
            headless=True,
            observation_type="accessibility_tree", 
            current_viewport_only=True,
            viewport_size={"width": 1280, "height": 720}
        )
        self.evaluator = Evaluator()
        
        # Initialize monitoring
        self.cost_monitor = WebArenaCostMonitor()
        self.comm_monitor = CommunicationMonitor()
        self.action_monitor = ActionMonitor()
        
    def evaluate_mas(self, mas_system, task_configs: List[Dict]):
        """Evaluate MAS on WebArena tasks"""
        results = []
        
        for task_config in task_configs:
            result = self.run_single_task(mas_system, task_config)
            results.append(result)
            
        return self.aggregate_results(results)
    
    def run_single_task(self, mas_system, task_config):
        """Run one WebArena task with monitoring"""
        # Reset environment and monitors
        obs = self.env.reset(task_config)
        self.reset_monitors()
        
        # Initialize task
        task = {
            'goal': task_config['goal'],
            'start_url': task_config['start_url'],
            'observation': obs
        }
        
        # Track metrics
        trajectory = []
        total_cost = 0
        start_time = time.time()
        
        # Run until done or max steps
        done = False
        step = 0
        max_steps = task_config.get('max_steps', 30)
        
        while not done and step < max_steps:
            # MAS decides next action
            mas_response = mas_system.solve_web_step(
                goal=task['goal'],
                current_observation=obs,
                previous_actions=trajectory
            )
            
            # Extract action and track costs
            action = self.parse_mas_response(mas_response)
            step_cost = self.calculate_step_cost(mas_response)
            total_cost += step_cost
            
            # Log the action
            self.action_monitor.log_action(action, step_cost)
            
            # Execute action in environment
            try:
                obs, reward, done, info = self.env.step(action)
            except Exception as e:
                # Handle failed actions
                obs = f"Error executing action: {str(e)}"
                done = False
                
            # Record trajectory
            trajectory.append({
                'step': step,
                'action': action,
                'observation': obs,
                'cost': step_cost,
                'agent_used': mas_response.agent_id,
                'communications': self.comm_monitor.get_step_messages()
            })
            
            step += 1
        
        # Evaluate final result
        success = self.evaluator.evaluate(
            task_config=task_config,
            trajectory=trajectory,
            final_state=self.env.get_state()
        )
        
        return {
            'task_id': task_config['task_id'],
            'success': success,
            'total_cost': total_cost,
            'steps': len(trajectory),
            'time_taken': time.time() - start_time,
            'trajectory': trajectory,
            'communication_stats': self.comm_monitor.get_stats(),
            'cost_breakdown': self.cost_monitor.get_breakdown()
        }
    
    def reset_monitors(self):
        """Reset all monitors for new task"""
        self.comm_monitor.reset()
        self.action_monitor.reset()
        
    def parse_mas_response(self, mas_response):
        """Extract action from MAS response"""
        if hasattr(mas_response, 'action'):
            return mas_response.action
        elif isinstance(mas_response, dict):
            return mas_response.get('action', {})
        else:
            return {'type': 'wait', 'text': str(mas_response)}
            
    def calculate_step_cost(self, mas_response):
        """Calculate cost of this step"""
        # Base cost per step
        cost = 0.001
        
        # Add LLM costs if available
        if hasattr(mas_response, 'tokens_used'):
            cost += mas_response.tokens_used * 0.00003  # Rough GPT-4 pricing
            
        # Add vision costs if screenshot was analyzed
        if hasattr(mas_response, 'used_vision'):
            cost += 0.01 if mas_response.used_vision else 0
            
        return cost
        
    def aggregate_results(self, results):
        """Aggregate results across all tasks"""
        if not results:
            return {'success_rate': 0, 'avg_cost': 0, 'avg_steps': 0}
            
        total_tasks = len(results)
        successes = sum(1 for r in results if r['success'])
        total_cost = sum(r['total_cost'] for r in results)
        total_steps = sum(r['steps'] for r in results)
        
        return {
            'success_rate': successes / total_tasks,
            'avg_cost': total_cost / total_tasks,
            'avg_steps': total_steps / total_tasks,
            'results': results
        }