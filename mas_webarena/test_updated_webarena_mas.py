#!/usr/bin/env python3
"""Test the updated WebArenaMAS as specified in CLAUDE.md"""

from WebArenaMAS import WebArenaMAS
from utils.mas_response import MASResponse


class MockAgent:
    """Mock agent for testing"""
    def __init__(self, agent_id, behavior='normal'):
        self.id = agent_id
        self.behavior = behavior
        
    def generate_action(self, state, allowed_actions):
        """Generate mock action based on behavior"""
        if self.behavior == 'normal':
            # Check if there are buttons available
            if state['current_page'].get('has_button'):
                return {
                    'action': {'type': 'click', 'element_id': 1},
                    'reasoning': f"Agent {self.id} found a button to click",
                    'confidence': 0.8,
                    'tokens_used': 50
                }
            else:
                return {
                    'action': {'type': 'wait'},
                    'reasoning': f"Agent {self.id} waiting for better opportunity",
                    'confidence': 0.4,
                    'tokens_used': 20
                }
        elif self.behavior == 'vision':
            return {
                'action': {'type': 'type', 'text': 'search term'},
                'reasoning': f"Agent {self.id} using vision to analyze page",
                'confidence': 0.9,
                'tokens_used': 100,
                'used_vision': True
            }
        elif self.behavior == 'expensive':
            return {
                'action': {'type': 'scroll', 'direction': 'down'},
                'reasoning': f"Agent {self.id} performing complex analysis",
                'confidence': 0.7,
                'tokens_used': 1000
            }


class MockOrchestrator:
    """Mock orchestrator for testing"""
    def __init__(self, agents):
        self.agents = agents
        self.selection_strategy = 'round_robin'
        self.current_index = 0
        
    def select_agent(self, state):
        """Select agent based on strategy"""
        if self.selection_strategy == 'round_robin':
            agent = self.agents[self.current_index % len(self.agents)]
            self.current_index += 1
            return agent
        elif self.selection_strategy == 'budget_aware':
            # Select cheaper agents when budget is low
            if state['remaining_budget'] < 0.1:
                # Find agent with lowest cost behavior
                return min(self.agents, key=lambda a: 0 if a.behavior == 'normal' else 1)
            else:
                return self.agents[0]  # Default to first agent


def test_basic_mas_functionality():
    """Test basic WebArenaMAS functionality with updated components"""
    print("Testing basic WebArenaMAS functionality...")
    
    # Create mock agents and orchestrator
    agents = [
        MockAgent("navigation_agent", "normal"),
        MockAgent("vision_agent", "vision"),
        MockAgent("analysis_agent", "expensive")
    ]
    orchestrator = MockOrchestrator(agents)
    
    # Initialize MAS with budget
    mas = WebArenaMAS(agents, orchestrator, initial_budget=0.5)
    
    assert mas.budget_tracker.initial_budget == 0.5, "Should initialize with correct budget"
    assert mas.budget_tracker.remaining == 0.5, "Should start with full budget"
    
    # Test with observation that has buttons
    observation_with_buttons = """[1] <button text="Search" clickable>
[2] <input type="text" text="" placeholder="Enter search term">
[3] <button text="Submit" clickable>"""
    
    goal = "Find and click the search button"
    previous_actions = []
    
    # First step
    response1 = mas.solve_web_step(goal, observation_with_buttons, previous_actions)
    
    assert isinstance(response1, MASResponse), "Should return MASResponse object"
    assert response1.agent_id == "navigation_agent", "Should use navigation agent first"
    assert response1.action['type'] == 'click', "Should choose click action when button available"
    assert response1.confidence > 0, "Should have positive confidence"
    assert mas.budget_tracker.remaining < 0.5, "Should consume budget"
    
    print(f"Step 1 - Agent: {response1.agent_id}, Action: {response1.action}, Budget remaining: ${mas.budget_tracker.remaining:.4f}")
    
    # Second step with vision agent
    response2 = mas.solve_web_step(goal, observation_with_buttons, [response1])
    
    assert response2.agent_id == "vision_agent", "Should use vision agent second"
    assert response2.used_vision == True, "Vision agent should use vision"
    assert response2.tokens_used > response1.tokens_used, "Vision agent should use more tokens"
    
    print(f"Step 2 - Agent: {response2.agent_id}, Action: {response2.action}, Vision: {response2.used_vision}, Budget remaining: ${mas.budget_tracker.remaining:.4f}")
    
    print("✅ Basic MAS functionality tests passed!")


def test_budget_tracking_integration():
    """Test budget tracking integration with MAS"""
    print("\nTesting budget tracking integration...")
    
    agents = [MockAgent("test_agent", "expensive")]
    orchestrator = MockOrchestrator(agents)
    
    # Start with small budget
    mas = WebArenaMAS(agents, orchestrator, initial_budget=0.05)
    
    observation = """[1] <div text="Simple page">"""
    goal = "Navigate the page"
    
    initial_budget = mas.budget_tracker.remaining
    print(f"Initial budget: ${initial_budget:.4f}")
    
    # Execute action that should consume significant budget
    response = mas.solve_web_step(goal, observation, [])
    
    remaining_budget = mas.budget_tracker.remaining
    consumed = initial_budget - remaining_budget
    
    print(f"Action cost: ${consumed:.4f}, Remaining: ${remaining_budget:.4f}")
    
    assert consumed > 0, "Should consume budget"
    assert response.tokens_used == 1000, "Should track high token usage"
    
    # Check cost calculation
    expected_cost = 0.001 + (1000 * 0.00003)  # base + token cost
    actual_cost = consumed
    assert abs(actual_cost - expected_cost) < 0.001, f"Expected ${expected_cost:.4f}, got ${actual_cost:.4f}"
    
    print("✅ Budget tracking integration tests passed!")


def test_observation_processing_integration():
    """Test observation processing integration"""
    print("\nTesting observation processing integration...")
    
    agents = [MockAgent("test_agent", "normal")]
    orchestrator = MockOrchestrator(agents)
    mas = WebArenaMAS(agents, orchestrator)
    
    # Test with complex observation
    complex_observation = """[1] <form>
[2] <input type="email" text="" placeholder="Email">
[3] <input type="password" text="" placeholder="Password">
[4] <button text="Login" clickable>
[5] <a href="/forgot" text="Forgot password?" clickable>"""
    
    goal = "Login to the website"
    
    response = mas.solve_web_step(goal, complex_observation, [])
    
    # Verify the observation was properly processed
    # The agent should detect buttons and forms through the processed observation
    assert response.action['type'] == 'click', "Should choose click action with forms and buttons present"
    assert response.reasoning.find("button") != -1, "Reasoning should mention button detection"
    
    print(f"Complex observation processed - Action: {response.action}, Reasoning: {response.reasoning}")
    
    # Test with error observation
    error_observation = "Error: Page not found (404)"
    
    error_response = mas.solve_web_step(goal, error_observation, [response])
    
    # With error, agent should wait since no actionable elements
    assert error_response.action['type'] == 'wait', "Should wait when encountering errors"
    
    print(f"Error observation handled - Action: {error_response.action}")
    
    print("✅ Observation processing integration tests passed!")


def test_mas_reset_functionality():
    """Test MAS reset functionality"""
    print("\nTesting MAS reset functionality...")
    
    agents = [MockAgent("test_agent", "vision")]
    orchestrator = MockOrchestrator(agents)
    mas = WebArenaMAS(agents, orchestrator, initial_budget=1.0)
    
    # Use some budget
    observation = """[1] <button text="Test" clickable>"""
    response = mas.solve_web_step("test goal", observation, [])
    
    initial_remaining = mas.budget_tracker.remaining
    assert initial_remaining < 1.0, "Should have consumed some budget"
    
    print(f"Before reset - Budget remaining: ${initial_remaining:.4f}")
    
    # Reset without new budget
    mas.reset()
    
    assert mas.budget_tracker.remaining == 1.0, "Should reset to original budget"
    assert mas.budget_tracker.spent == 0.0, "Should reset spent amount"
    assert len(mas.budget_tracker.history) == 0, "Should clear history"
    
    print(f"After reset - Budget remaining: ${mas.budget_tracker.remaining:.4f}")
    
    # Reset with new budget
    mas.reset(budget=2.0)
    
    assert mas.budget_tracker.initial_budget == 2.0, "Should update initial budget"
    assert mas.budget_tracker.remaining == 2.0, "Should reset to new budget"
    
    print(f"After reset with new budget - Budget remaining: ${mas.budget_tracker.remaining:.4f}")
    
    print("✅ MAS reset functionality tests passed!")


def test_cost_estimation():
    """Test action cost estimation logic"""
    print("\nTesting cost estimation logic...")
    
    agents = [MockAgent("test_agent", "normal")]
    orchestrator = MockOrchestrator(agents)
    mas = WebArenaMAS(agents, orchestrator)
    
    # Test different response types and their costs
    test_responses = [
        {'tokens_used': 0, 'used_vision': False},  # Base cost only
        {'tokens_used': 100, 'used_vision': False},  # Base + tokens
        {'tokens_used': 0, 'used_vision': True},  # Base + vision
        {'tokens_used': 200, 'used_vision': True},  # Base + tokens + vision
    ]
    
    expected_costs = [
        0.001,  # Base only
        0.001 + (100 * 0.00003),  # Base + token cost
        0.001 + 0.01,  # Base + vision cost
        0.001 + (200 * 0.00003) + 0.01  # All costs
    ]
    
    for i, response in enumerate(test_responses):
        cost = mas._estimate_action_cost(response)
        expected = expected_costs[i]
        assert abs(cost - expected) < 0.00001, f"Test {i}: Expected ${expected:.5f}, got ${cost:.5f}"
        print(f"Cost test {i+1}: ${cost:.5f} (tokens: {response.get('tokens_used', 0)}, vision: {response.get('used_vision', False)})")
    
    print("✅ Cost estimation tests passed!")


if __name__ == "__main__":
    test_basic_mas_functionality()
    test_budget_tracking_integration()
    test_observation_processing_integration()
    test_mas_reset_functionality()
    test_cost_estimation()
    print("\n🎉 All updated WebArenaMAS integration tests passed successfully!")