class WebArenaCostMonitor:
    """Track costs specific to web automation"""
    
    def __init__(self):
        self.costs = {
            'llm_calls': 0,
            'vision_api_calls': 0,  # For screenshot analysis
            'action_executions': 0,
            'total_tokens': 0
        }
        
    def track_llm_call(self, model, tokens_in, tokens_out):
        """Track LLM API costs"""
        model_costs = {
            'gpt-4': {'input': 0.03/1000, 'output': 0.06/1000},
            'gpt-4-vision': {'input': 0.01/1000, 'output': 0.03/1000},
            'claude-3': {'input': 0.015/1000, 'output': 0.075/1000}
        }
        
        if model in model_costs:
            cost = (model_costs[model]['input'] * tokens_in + 
                    model_costs[model]['output'] * tokens_out)
            self.costs['llm_calls'] += cost
            self.costs['total_tokens'] += tokens_in + tokens_out
            
    def track_vision_analysis(self, screenshot_size):
        """Track cost of analyzing screenshots"""
        # Approximate cost based on image size
        cost = screenshot_size / (1024 * 1024) * 0.01  # $0.01 per MB
        self.costs['vision_api_calls'] += cost
    
    def track_action_execution(self, action_type, complexity_score=1):
        """Track cost of executing web actions"""
        # Base costs per action type
        action_costs = {
            'click': 0.001,
            'type': 0.002,
            'scroll': 0.0005,
            'navigate': 0.003,
            'wait': 0.0001,
            'verify': 0.002
        }
        
        base_cost = action_costs.get(action_type, 0.001)
        total_cost = base_cost * complexity_score
        self.costs['action_executions'] += total_cost
        
        return total_cost
    
    def get_cost_breakdown(self):
        """Get detailed cost breakdown"""
        return {
            'llm_calls': self.costs['llm_calls'],
            'vision_api_calls': self.costs['vision_api_calls'],
            'action_executions': self.costs['action_executions'],
            'total_cost': sum(self.costs.values()),
            'total_tokens': self.costs['total_tokens']
        }
    
    def get_cost_per_task(self, task_id, task_actions):
        """Calculate cost for a specific task"""
        task_cost = 0
        for action in task_actions:
            if 'cost' in action:
                task_cost += action['cost']
        return task_cost
    
    def estimate_future_costs(self, remaining_actions, avg_action_cost=None):
        """Estimate costs for remaining actions in a task"""
        if avg_action_cost is None:
            completed_actions = self.costs['action_executions']
            # Rough estimate based on current average
            avg_action_cost = completed_actions / max(1, len(self.get_cost_breakdown()))
        
        return remaining_actions * avg_action_cost
    
    def get_cost_efficiency_metrics(self):
        """Calculate cost efficiency metrics"""
        total_cost = sum(self.costs.values())
        
        if total_cost == 0:
            return {}
        
        return {
            'cost_per_token': total_cost / max(1, self.costs['total_tokens']),
            'llm_cost_ratio': self.costs['llm_calls'] / total_cost,
            'vision_cost_ratio': self.costs['vision_api_calls'] / total_cost,
            'action_cost_ratio': self.costs['action_executions'] / total_cost,
            'total_cost': total_cost
        }

        