from utils.observation_processor import ObservationProcessor
from utils.mas_response import MASResponse
from utils.budget_tracker import BudgetTracker

class WebArenaMAS:
    
    def __init__(self, agents, orchestrator):
        self.agents = agents
        self.orchestrator = orchestrator
        self.observation_processor = ObservationProcessor()
        
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
        # Different agents might be good at different things:
        # - NavigationAgent: good at finding elements
        # - FormFillerAgent: good at completing forms
        # - VerificationAgent: good at checking results
        
        response = selected_agent.generate_action(
            state,
            allowed_actions=['click', 'type', 'select', 'scroll', 'back']
        )
        
        return MASResponse(
            agent_id=selected_agent.id,
            action=response.action,
            reasoning=response.reasoning,
            confidence=response.confidence
        )