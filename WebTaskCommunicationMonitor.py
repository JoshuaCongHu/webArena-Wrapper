class WebTaskCommunicationMonitor:
    """Monitor inter-agent communication during web tasks"""
    
    def analyze_communication_patterns(self, trajectory):
        patterns = {
            'navigation_help_requests': 0,  # "I can't find the button"
            'state_verifications': 0,       # "Can you confirm we're on checkout?"
            'strategy_discussions': 0,      # "Should we try search instead?"
            'error_consultations': 0        # "Action failed, what now?"
        }
        
        for comm in self.communication_log:
            if self.is_navigation_help(comm):
                patterns['navigation_help_requests'] += 1
            # ... etc
            
        return patterns
    

    