import time

class ActionMonitor:
    """Monitor agent actions and their costs"""
    
    def __init__(self):
        self.actions = []
        
    def log_action(self, action, cost):
        """Log an action taken by an agent"""
        action_record = {
            'action': action,
            'cost': cost,
            'timestamp': time.time()
        }
        self.actions.append(action_record)
        
    def get_action_stats(self):
        """Get action statistics"""
        if not self.actions:
            return {'total_actions': 0, 'total_cost': 0, 'avg_cost': 0}
            
        total_cost = sum(a['cost'] for a in self.actions)
        return {
            'total_actions': len(self.actions),
            'total_cost': total_cost,
            'avg_cost': total_cost / len(self.actions),
            'action_types': self._count_action_types()
        }
        
    def _count_action_types(self):
        """Count different types of actions"""
        types = {}
        for action in self.actions:
            action_type = action['action'].get('type', 'unknown')
            types[action_type] = types.get(action_type, 0) + 1
        return types
        
    def reset(self):
        """Reset for new task"""
        self.actions = []