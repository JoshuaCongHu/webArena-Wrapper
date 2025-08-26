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