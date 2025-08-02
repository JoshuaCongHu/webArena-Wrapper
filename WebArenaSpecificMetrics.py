class WebArenaSpecificMetrics:
    """Metrics specific to web automation"""
    
    def calculate_efficiency_metrics(self, trajectory):
        return {
            # Web-specific metrics
            'actions_per_task': len(trajectory),
            'failed_actions': sum(1 for t in trajectory if t.get('error')),
            'backtrack_rate': self.count_backtracks(trajectory),
            'dom_queries': sum(1 for t in trajectory if t['action']['type'] == 'query'),
            
            # Cost breakdown
            'navigation_cost': sum(t['cost'] for t in trajectory if 'click' in t['action']),
            'form_filling_cost': sum(t['cost'] for t in trajectory if 'type' in t['action']),
            'verification_cost': sum(t['cost'] for t in trajectory if 'verify' in t['action']),
            
            # Time metrics
            'time_per_action': self.calculate_avg_time(trajectory),
            'idle_time': self.calculate_idle_time(trajectory)
        }
    
    