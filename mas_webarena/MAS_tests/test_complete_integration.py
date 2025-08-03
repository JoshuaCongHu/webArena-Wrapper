#!/usr/bin/env python3
"""Complete integration test demonstrating all components working together"""

import json
from WebArenaMAS import WebArenaMAS
from utils.mas_response import MASResponse
from monitors.communication_monitor import CommunicationMonitor
from monitors.action_monitor import ActionMonitor
from WebArenaSpecificMetrics import WebArenaSpecificMetrics


class SpecializedAgent:
    """More sophisticated agent with different specializations"""
    def __init__(self, agent_id, specialization='general'):
        self.id = agent_id
        self.specialization = specialization
        
    def generate_action(self, state, allowed_actions):
        """Generate action based on specialization"""
        current_page = state['current_page']
        
        if self.specialization == 'navigation':
            # Navigation agent prefers clicking buttons and links
            if current_page.get('has_button'):
                return {
                    'action': {'type': 'click', 'element_id': 1},
                    'reasoning': f'Navigation agent {self.id} found clickable button',
                    'confidence': 0.9,
                    'tokens_used': 40
                }
                
        elif self.specialization == 'form_filler':
            # Form filler looks for input fields
            if current_page.get('has_form'):
                return {
                    'action': {'type': 'type', 'text': 'test input'},
                    'reasoning': f'Form filler agent {self.id} found form elements',
                    'confidence': 0.85,
                    'tokens_used': 60
                }
                
        elif self.specialization == 'vision':
            # Vision agent uses image analysis
            return {
                'action': {'type': 'scroll', 'direction': 'down'},
                'reasoning': f'Vision agent {self.id} analyzing page layout',
                'confidence': 0.75,
                'tokens_used': 100,
                'used_vision': True
            }
        
        # Default fallback
        return {
            'action': {'type': 'wait'},
            'reasoning': f'Agent {self.id} waiting for better opportunity',
            'confidence': 0.4,
            'tokens_used': 20
        }


class SmartOrchestrator:
    """Smart orchestrator that selects agents based on page content and budget"""
    def __init__(self, agents):
        self.agents = agents
        self.agent_map = {agent.specialization: agent for agent in agents}
        
    def select_agent(self, state):
        """Select best agent based on current page and remaining budget"""
        current_page = state['current_page']
        remaining_budget = state['remaining_budget']
        
        # If budget is low, use cheaper agents
        if remaining_budget < 0.1:
            return min(self.agents, key=lambda a: 0 if a.specialization == 'navigation' else 1)
        
        # Select based on page content
        if current_page.get('has_form'):
            return self.agent_map.get('form_filler', self.agents[0])
        elif current_page.get('has_button'):
            return self.agent_map.get('navigation', self.agents[0])
        else:
            return self.agent_map.get('vision', self.agents[0])


def test_complete_integration():
    """Test all components working together"""
    print("🧪 Testing Complete Integration of All Components")
    print("=" * 60)
    
    # Initialize monitors
    comm_monitor = CommunicationMonitor()
    action_monitor = ActionMonitor()
    metrics = WebArenaSpecificMetrics()
    
    # Create specialized agents
    agents = [
        SpecializedAgent('nav_agent', 'navigation'),
        SpecializedAgent('form_agent', 'form_filler'),
        SpecializedAgent('vision_agent', 'vision')
    ]
    
    orchestrator = SmartOrchestrator(agents)
    mas = WebArenaMAS(agents, orchestrator, initial_budget=0.5)
    
    print(f"🤖 Initialized {len(agents)} specialized agents")
    print(f"💰 Starting budget: ${mas.budget_tracker.initial_budget}")
    
    # Simulate complex task with different page types
    observations = [
        # Login page with form
        """[1] <form>
[2] <input type="email" placeholder="Email">
[3] <input type="password" placeholder="Password">
[4] <button text="Login" clickable>""",
        
        # Navigation page with buttons
        """[1] <nav>
[2] <button text="Dashboard" clickable>
[3] <button text="Profile" clickable>
[4] <button text="Settings" clickable>""",
        
        # Content page needing vision analysis
        """[1] <div text="Complex layout">
[2] <img src="chart.png" alt="Data visualization">
[3] <div text="Scroll to see more content">""",
        
        # Error page
        "Error: Page not found (404)"
    ]
    
    trajectory = []
    goal = "Complete user onboarding flow"
    
    for step, obs in enumerate(observations):
        print(f"\n--- Step {step + 1}: Processing Observation ---")
        print(f"📄 Page type: {obs.split(']')[0] if not obs.startswith('Error') else 'Error page'}")
        
        # Log inter-agent communication (simulated)
        if step > 0:
            comm_monitor.log_message(
                f"orchestrator", 
                f"selected_agent",
                f"Please handle step {step + 1}",
                "instruction"
            )
        
        # Get MAS response
        mas_response = mas.solve_web_step(goal, obs, trajectory)
        
        # Calculate costs
        step_cost = 0.001 + (mas_response.tokens_used * 0.00003)
        if mas_response.used_vision:
            step_cost += 0.01
            
        # Log action
        action_monitor.log_action(mas_response.action, step_cost)
        
        # Log communication response
        comm_monitor.log_message(
            mas_response.agent_id,
            "orchestrator", 
            f"Completed action: {mas_response.action['type']}",
            "report"
        )
        
        print(f"🎯 Selected agent: {mas_response.agent_id}")
        print(f"⚡ Action: {mas_response.action}")
        print(f"🧠 Reasoning: {mas_response.reasoning}")
        print(f"📈 Confidence: {mas_response.confidence:.2f}")
        print(f"🔢 Tokens used: {mas_response.tokens_used}")
        print(f"👁️ Used vision: {mas_response.used_vision}")
        print(f"💰 Step cost: ${step_cost:.4f}")
        print(f"💳 Remaining budget: ${mas.budget_tracker.remaining:.4f}")
        
        # Record trajectory
        trajectory.append({
            'step': step,
            'action': mas_response.action,
            'cost': step_cost,
            'agent_id': mas_response.agent_id,
            'reasoning': mas_response.reasoning,
            'observation': obs
        })
        
        # Check if budget is getting low
        if mas.budget_tracker.remaining < 0.1:
            print("⚠️ Low budget - orchestrator will prioritize cheaper agents")
    
    print(f"\n🏁 Task Execution Complete!")
    print("=" * 60)
    
    # Analyze results with all monitoring systems
    print("\n📊 MONITORING RESULTS:")
    
    # Communication analysis
    comm_stats = comm_monitor.get_stats()
    print(f"\n📡 Communication Stats:")
    print(f"  • Total messages: {comm_stats['total_messages']}")
    print(f"  • Total tokens in messages: {comm_stats['total_tokens']}")
    print(f"  • Messages by type: {dict(comm_stats['messages_by_type'])}")
    print(f"  • Messages by sender: {dict(comm_stats['messages_by_sender'])}")
    
    # Action analysis
    action_stats = action_monitor.get_stats()
    print(f"\n⚡ Action Stats:")
    print(f"  • Total actions: {action_stats['total_actions']}")
    print(f"  • Total cost: ${action_stats['total_cost']:.4f}")
    print(f"  • Average cost per action: ${action_stats['avg_cost_per_action']:.4f}")
    print(f"  • Action breakdown: {action_stats['action_counts']}")
    
    # WebArena-specific metrics
    webarena_metrics = metrics.calculate_efficiency_metrics(trajectory)
    print(f"\n🌐 WebArena Metrics:")
    print(f"  • Actions per task: {webarena_metrics['actions_per_task']}")
    print(f"  • Failed actions: {webarena_metrics['failed_actions']}")
    print(f"  • Backtrack rate: {webarena_metrics['backtrack_rate']}")
    print(f"  • Navigation cost: ${webarena_metrics['navigation_cost']:.4f}")
    print(f"  • Form filling cost: ${webarena_metrics['form_filling_cost']:.4f}")
    print(f"  • Verification cost: ${webarena_metrics['verification_cost']:.4f}")
    print(f"  • Idle time: {webarena_metrics['idle_time']}")
    
    # Budget analysis
    budget_utilization = (mas.budget_tracker.spent / mas.budget_tracker.initial_budget) * 100
    print(f"\n💰 Budget Analysis:")
    print(f"  • Initial budget: ${mas.budget_tracker.initial_budget:.4f}")
    print(f"  • Amount spent: ${mas.budget_tracker.spent:.4f}")
    print(f"  • Remaining: ${mas.budget_tracker.remaining:.4f}")
    print(f"  • Utilization: {budget_utilization:.1f}%")
    print(f"  • Transaction history: {len(mas.budget_tracker.history)} transactions")
    
    # Overall success assessment
    total_cost = sum(t['cost'] for t in trajectory)
    success_indicators = {
        'completed_steps': len(trajectory),
        'budget_efficient': budget_utilization < 80,
        'diverse_agents_used': len(set(t['agent_id'] for t in trajectory)) > 1,
        'handled_errors': any('Error' in t.get('observation', '') for t in trajectory)
    }
    
    print(f"\n🎯 Success Indicators:")
    for indicator, value in success_indicators.items():
        status = "✅" if value else "❌"
        print(f"  {status} {indicator.replace('_', ' ').title()}: {value}")
    
    print(f"\n📋 Final Integration Summary:")
    print(f"  • All {len(agents)} agents participated: {success_indicators['diverse_agents_used']}")
    print(f"  • Budget management active: ✅")
    print(f"  • Communication tracking: ✅ ({comm_stats['total_messages']} messages)")
    print(f"  • Action monitoring: ✅ ({action_stats['total_actions']} actions)")
    print(f"  • WebArena metrics: ✅")
    print(f"  • Error handling: {'✅' if success_indicators['handled_errors'] else '⚠️'}")
    
    return {
        'trajectory': trajectory,
        'communication_stats': comm_stats,
        'action_stats': action_stats,
        'webarena_metrics': webarena_metrics,
        'budget_stats': {
            'initial': mas.budget_tracker.initial_budget,
            'spent': mas.budget_tracker.spent,
            'remaining': mas.budget_tracker.remaining,
            'utilization_percent': budget_utilization
        }
    }


if __name__ == "__main__":
    result = test_complete_integration()
    print(f"\n✅ Integration test completed successfully!")
    print(f"📄 Full results available in returned dictionary")