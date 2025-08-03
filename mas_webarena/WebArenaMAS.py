from utils.observation_processor import ObservationProcessor
from utils.budget_tracker import BudgetTracker
from utils.mas_response import MASResponse

class WebArenaMAS:
    """MAS system adapted for WebArena tasks"""
    
    def __init__(self, agents, orchestrator, initial_budget=1.0):
        self.agents = agents
        self.orchestrator = orchestrator
        self.observation_processor = ObservationProcessor()
        self.budget_tracker = BudgetTracker(initial_budget)
        
    def solve_web_step(self, goal, current_observation, previous_actions):
        """Solve one step of a web task"""
        
        # Process observation (HTML/screenshot -> structured data)
        processed_obs = self.observation_processor.process(current_observation)
        
        # Create state representation
        state = {
            'goal': goal,
            'current_page': processed_obs,
            'action_history': previous_actions,
            'remaining_budget': self.budget_tracker.remaining
        }
        
        # Orchestrator selects agent based on state
        selected_agent = self.orchestrator.select_agent(state)
        
        # Agent generates action
        response = selected_agent.generate_action(
            state,
            allowed_actions=['click', 'type', 'select', 'scroll', 'back', 'wait']
        )
        
        # Track budget consumption
        estimated_cost = self._estimate_action_cost(response)
        self.budget_tracker.consume(estimated_cost)
        
        return MASResponse(
            agent_id=selected_agent.id,
            action=response.get('action', {'type': 'wait'}),
            reasoning=response.get('reasoning', ''),
            confidence=response.get('confidence', 0.5),
            tokens_used=response.get('tokens_used', 0),
            used_vision=response.get('used_vision', False)
        )
        
    def _estimate_action_cost(self, response):
        """Estimate cost of an action"""
        base_cost = 0.001
        if response.get('tokens_used'):
            base_cost += response['tokens_used'] * 0.00003
        if response.get('used_vision'):
            base_cost += 0.01
        return base_cost
        
    def reset(self, budget=None):
        """Reset MAS for new task"""
        self.budget_tracker.reset(budget)