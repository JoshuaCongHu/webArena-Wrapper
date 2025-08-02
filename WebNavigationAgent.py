class WebNavigationAgent:
    """Agent specialized for web navigation"""
    
    def generate_action(self, state, allowed_actions):
        # Use LLM to understand page and generate action
        prompt = f"""
        Goal: {state['goal']}
        Current page elements: {state['current_page']['elements']}
        Previous actions: {state['action_history'][-3:]}
        
        What's the next action to achieve the goal?
        """
        
        response = self.llm.generate(prompt)
        action = self.parse_action(response)
        
        # Track cost
        self.cost_tracker.track_llm_call(
            model=self.model_name,
            tokens_in=count_tokens(prompt),
            tokens_out=count_tokens(response)
        )
        
        return action
    
    