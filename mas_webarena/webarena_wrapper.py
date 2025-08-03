# webarena_wrapper.py
import time
from typing import Dict, List, Any
# Fix WebArena imports - use actual WebArena structure
from browser_env import ScriptBrowserEnv
from evaluation_harness.evaluators import evaluator_router

# Import your monitors
from monitors.communication_monitor import CommunicationMonitor
from monitors.action_monitor import ActionMonitor
from webArenaCostMonitor import WebArenaCostMonitor

class WebArenaWrapper:
    """Wrapper for WebArena that adds cost and communication monitoring"""
    
    def __init__(self, env_config=None):
        # Use actual WebArena environment
        self.env = ScriptBrowserEnv(
            headless=True,
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": 1280, "height": 720}
        )
        
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