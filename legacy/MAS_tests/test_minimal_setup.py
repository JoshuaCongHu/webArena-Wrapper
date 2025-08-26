import json
from WebArenaMAS import WebArenaMAS
from utils.mas_response import MASResponse

class MinimalAgent:
    """Minimal agent for testing"""
    def __init__(self, agent_id='test_agent'):
        self.id = agent_id
        
    def generate_action(self, state, allowed_actions):
        """Generate simple action based on state"""
        # Very basic logic for testing
        if state['current_page'].get('has_button'):
            return {
                'action': {'type': 'click', 'element_id': 0},
                'reasoning': 'Found a button to click',
                'confidence': 0.8,
                'tokens_used': 50
            }
        return {
            'action': {'type': 'wait'},
            'reasoning': 'No actionable elements found',
            'confidence': 0.3,
            'tokens_used': 30
        }

class MinimalOrchestrator:
    """Minimal orchestrator for testing"""
    def __init__(self, agents):
        self.agents = agents
        
    def select_agent(self, state):
        """Select first agent for now"""
        return self.agents[0]

# Mock WebArenaWrapper for testing
class MockWebArenaWrapper:
    """Mock wrapper for testing minimal setup"""
    
    def run_single_task(self, mas_system, task_config):
        """Simulate running a single task"""
        # Mock observations that would come from WebArena
        mock_observations = [
            """[1] <div text="Welcome to the website">
[2] <button text="Search" clickable>
[3] <input type="text" placeholder="Search term">""",
            """[1] <div text="Search results loading...">
[2] <button text="Back" clickable>""",
            """[1] <div text="Search completed">
[2] <a href="/results" text="View Results" clickable>"""
        ]
        
        trajectory = []
        total_cost = 0
        
        print(f"\n🚀 Starting task: {task_config['goal']}")
        print(f"📍 URL: {task_config['start_url']}")
        print(f"📊 Max steps: {task_config['max_steps']}")
        
        for step in range(min(len(mock_observations), task_config['max_steps'])):
            current_obs = mock_observations[step]
            
            print(f"\n--- Step {step + 1} ---")
            print(f"🔍 Observation: {current_obs[:100]}...")
            
            # Get MAS response
            mas_response = mas_system.solve_web_step(
                goal=task_config['goal'],
                current_observation=current_obs,
                previous_actions=trajectory
            )
            
            # Simulate cost calculation
            step_cost = 0.001 + (mas_response.tokens_used * 0.00003)
            if mas_response.used_vision:
                step_cost += 0.01
            total_cost += step_cost
            
            print(f"🤖 Agent: {mas_response.agent_id}")
            print(f"⚡ Action: {mas_response.action}")
            print(f"🧠 Reasoning: {mas_response.reasoning}")
            print(f"📈 Confidence: {mas_response.confidence:.2f}")
            print(f"💰 Step cost: ${step_cost:.4f}")
            
            # Record step
            trajectory.append({
                'step': step,
                'action': mas_response.action,
                'observation': current_obs,
                'cost': step_cost,
                'agent_id': mas_response.agent_id,
                'reasoning': mas_response.reasoning,
                'confidence': mas_response.confidence
            })
            
        # Simulate success evaluation (simple heuristic)
        success = any('click' in t['action'].get('type', '') for t in trajectory)
        
        result = {
            'task_id': task_config['task_id'],
            'success': success,
            'total_cost': total_cost,
            'steps': len(trajectory),
            'trajectory': trajectory,
            'final_reasoning': trajectory[-1]['reasoning'] if trajectory else 'No actions taken'
        }
        
        print(f"\n✅ Task completed!")
        print(f"🏆 Success: {success}")
        print(f"💰 Total cost: ${total_cost:.4f}")
        print(f"📝 Total steps: {len(trajectory)}")
        
        return result

# Test function
def test_wrapper():
    """Test the wrapper with minimal MAS"""
    wrapper = MockWebArenaWrapper()
    
    agents = [MinimalAgent()]
    orchestrator = MinimalOrchestrator(agents)
    mas = WebArenaMAS(agents, orchestrator)
    
    # Create simple test task
    test_task = {
        'task_id': 'test_001',
        'goal': 'Click the search button',
        'start_url': 'http://example.com',
        'max_steps': 5
    }
    
    result = wrapper.run_single_task(mas, test_task)
    print(f"\n📋 Final Result:")
    print(json.dumps(result, indent=2))
    
if __name__ == "__main__":
    test_wrapper()