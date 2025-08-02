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

        