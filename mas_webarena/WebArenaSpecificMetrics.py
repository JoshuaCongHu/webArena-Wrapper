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