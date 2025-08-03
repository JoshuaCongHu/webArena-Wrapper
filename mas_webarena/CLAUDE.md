# WebArena MAS Implementation Tasks

## Overview
Complete the WebArena wrapper implementation, keeping the wrapper minimal while building intelligence into the MAS system.

## Architecture Reminder
```
Wrapper (Simple Interface) → MAS System (All Intelligence)
- Load tasks                 - RL Policy
- Execute in environment     - Agent selection
- Track basic metrics        - Communication
- Return results             - Cost optimization
```

## Missing Components to Implement

### 1. Core Monitor Classes

#### `monitors/communication_monitor.py`
```python
import time
from typing import List, Dict, Any
from collections import defaultdict

class CommunicationMonitor:
    """Track inter-agent communication during task execution"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset for new task"""
        self.messages = []
        self.current_step_messages = []
        
    def log_message(self, sender_id: str, receiver_id: str, content: str, message_type: str = "general"):
        """Log a message between agents"""
        message = {
            'timestamp': time.time(),
            'sender': sender_id,
            'receiver': receiver_id,
            'content': content,
            'type': message_type,
            'tokens': self._count_tokens(content)
        }
        self.messages.append(message)
        self.current_step_messages.append(message)
        
    def get_step_messages(self):
        """Get messages from current step and clear buffer"""
        messages = self.current_step_messages.copy()
        self.current_step_messages = []
        return messages
        
    def get_stats(self):
        """Get communication statistics"""
        if not self.messages:
            return {'total_messages': 0, 'total_tokens': 0}
            
        stats = {
            'total_messages': len(self.messages),
            'total_tokens': sum(m['tokens'] for m in self.messages),
            'messages_by_type': defaultdict(int),
            'messages_by_sender': defaultdict(int)
        }
        
        for msg in self.messages:
            stats['messages_by_type'][msg['type']] += 1
            stats['messages_by_sender'][msg['sender']] += 1
            
        return dict(stats)
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count"""
        return len(text.split()) * 1.3  # Rough approximation
```

#### `monitors/action_monitor.py`
```python
import time
from typing import List, Dict, Any

class ActionMonitor:
    """Track actions taken during task execution"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset for new task"""
        self.actions = []
        self.action_counts = {
            'click': 0,
            'type': 0,
            'scroll': 0,
            'back': 0,
            'wait': 0,
            'select': 0
        }
        
    def log_action(self, action: Dict[str, Any], cost: float):
        """Log an action taken"""
        action_record = {
            'timestamp': time.time(),
            'action': action,
            'cost': cost,
            'type': action.get('type', 'unknown')
        }
        self.actions.append(action_record)
        
        # Update counts
        action_type = action.get('type', 'unknown')
        if action_type in self.action_counts:
            self.action_counts[action_type] += 1
            
    def get_stats(self):
        """Get action statistics"""
        total_cost = sum(a['cost'] for a in self.actions)
        return {
            'total_actions': len(self.actions),
            'total_cost': total_cost,
            'action_counts': self.action_counts.copy(),
            'avg_cost_per_action': total_cost / len(self.actions) if self.actions else 0
        }
```

### 2. Observation Processing

#### `utils/observation_processor.py`
```python
from typing import Dict, List, Any
import re
from bs4 import BeautifulSoup

class ObservationProcessor:
    """Process WebArena observations into structured data"""
    
    def process(self, observation: str) -> Dict[str, Any]:
        """Convert raw observation to structured format"""
        if isinstance(observation, str) and observation.startswith("Error"):
            return self._process_error(observation)
            
        # Process accessibility tree format
        return self._process_accessibility_tree(observation)
        
    def _process_accessibility_tree(self, obs: str) -> Dict[str, Any]:
        """Parse accessibility tree observation"""
        elements = []
        lines = obs.strip().split('\n') if isinstance(obs, str) else []
        
        for line in lines:
            element = self._parse_element_line(line)
            if element:
                elements.append(element)
                
        return {
            'type': 'accessibility_tree',
            'elements': elements,
            'num_elements': len(elements),
            'has_form': any(e['tag'] in ['input', 'textarea', 'select'] for e in elements),
            'has_button': any(e['tag'] == 'button' or 'button' in e.get('text', '').lower() for e in elements),
            'raw': obs
        }
        
    def _parse_element_line(self, line: str) -> Dict[str, Any]:
        """Parse a single element from accessibility tree"""
        # Basic parsing - adapt based on actual WebArena format
        element = {}
        
        # Extract element ID [number]
        id_match = re.search(r'\[(\d+)\]', line)
        if id_match:
            element['id'] = int(id_match.group(1))
            
        # Extract tag type
        tag_match = re.search(r'<(\w+)', line)
        if tag_match:
            element['tag'] = tag_match.group(1).lower()
            
        # Extract text content
        text_match = re.search(r'text="([^"]*)"', line)
        if text_match:
            element['text'] = text_match.group(1)
            
        # Extract other attributes
        element['clickable'] = 'clickable' in line.lower()
        element['type'] = re.search(r'type="([^"]*)"', line)
        
        return element if element else None
        
    def _process_error(self, error_obs: str) -> Dict[str, Any]:
        """Process error observations"""
        return {
            'type': 'error',
            'error_message': error_obs,
            'elements': [],
            'num_elements': 0
        }
```

### 3. MAS System Components

#### `utils/budget_tracker.py`
```python
class BudgetTracker:
    """Track budget consumption during task execution"""
    
    def __init__(self, initial_budget: float = 1.0):
        self.initial_budget = initial_budget
        self.remaining = initial_budget
        self.spent = 0.0
        self.history = []
        
    def consume(self, amount: float) -> bool:
        """Consume budget, return False if insufficient"""
        if amount > self.remaining:
            return False
        self.remaining -= amount
        self.spent += amount
        self.history.append({
            'amount': amount,
            'remaining': self.remaining,
            'timestamp': time.time()
        })
        return True
        
    def reset(self, budget: float = None):
        """Reset budget for new task"""
        self.initial_budget = budget or self.initial_budget
        self.remaining = self.initial_budget
        self.spent = 0.0
        self.history = []
```

#### `utils/mas_response.py`
```python
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class MASResponse:
    """Response from MAS system for a single step"""
    agent_id: str
    action: Dict[str, Any]
    reasoning: str
    confidence: float
    tokens_used: int = 0
    used_vision: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'action': self.action,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'tokens_used': self.tokens_used,
            'used_vision': self.used_vision
        }
```

### 4. Update WebArenaMAS

#### `mas/WebArenaMAS.py` (Updated)
```python
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
```

### 5. Implement WebArenaSpecificMetrics Methods

#### `metrics/WebArenaSpecificMetrics.py` (Updated)
```python
class WebArenaSpecificMetrics:
    """Metrics specific to web automation"""
    
    def calculate_efficiency_metrics(self, trajectory):
        metrics = {
            # Web-specific metrics
            'actions_per_task': len(trajectory),
            'failed_actions': sum(1 for t in trajectory if t.get('error')),
            'backtrack_rate': self.count_backtracks(trajectory),
            'dom_queries': sum(1 for t in trajectory if t['action'].get('type') == 'query'),
            
            # Cost breakdown
            'navigation_cost': sum(t['cost'] for t in trajectory if t['action'].get('type') == 'click'),
            'form_filling_cost': sum(t['cost'] for t in trajectory if t['action'].get('type') == 'type'),
            'verification_cost': sum(t['cost'] for t in trajectory if t['action'].get('type') in ['verify', 'wait']),
            
            # Time metrics
            'time_per_action': self.calculate_avg_time(trajectory),
            'idle_time': self.calculate_idle_time(trajectory)
        }
        return metrics
    
    def count_backtracks(self, trajectory):
        """Count how many times 'back' action was used"""
        return sum(1 for t in trajectory if t['action'].get('type') == 'back')
        
    def calculate_avg_time(self, trajectory):
        """Calculate average time between actions"""
        if len(trajectory) < 2:
            return 0.0
        # Assuming each trajectory item has a timestamp
        # This is a simplified version - adapt based on your needs
        return len(trajectory) / 30.0  # Rough estimate
        
    def calculate_idle_time(self, trajectory):
        """Calculate time spent on 'wait' actions"""
        return sum(1 for t in trajectory if t['action'].get('type') == 'wait')
```

### 6. Simple Test Implementation

#### `test_minimal_setup.py`
```python
import json
from webarena_wrapper import WebArenaWrapper
from mas.WebArenaMAS import WebArenaMAS
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

# Test function
def test_wrapper():
    """Test the wrapper with minimal MAS"""
    wrapper = WebArenaWrapper()
    
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
    print(json.dumps(result, indent=2))
    
if __name__ == "__main__":
    test_wrapper()
```

## Implementation Order

1. **First**: Implement the monitor classes (CommunicationMonitor, ActionMonitor)
2. **Second**: Implement ObservationProcessor and helper classes
3. **Third**: Update WebArenaMAS with budget tracking
4. **Fourth**: Complete WebArenaSpecificMetrics methods
5. **Finally**: Test with minimal setup

## Key Points

- The wrapper remains a simple interface to WebArena
- All intelligence (agent selection, RL, communication strategies) stays in the MAS system
- Monitors just track metrics, they don't make decisions
- Start with minimal implementations and gradually add complexity

## Next Steps After Implementation

1. Create simple baseline agents (random, rule-based)
2. Implement basic RL orchestrator
3. Add communication between agents
4. Run experiments comparing different strategies
5. Analyze cost-performance tradeoffs