import time

class CommunicationMonitor:
    """Monitor inter-agent communications"""
    
    def __init__(self):
        self.messages = []
        self.step_messages = []
        
    def log_message(self, from_agent, to_agent, message_type, content):
        """Log a communication between agents"""
        message = {
            'from': from_agent,
            'to': to_agent,
            'type': message_type,
            'content': content,
            'timestamp': time.time()
        }
        self.messages.append(message)
        self.step_messages.append(message)
        
    def get_step_messages(self):
        """Get messages from current step and reset"""
        messages = self.step_messages.copy()
        self.step_messages = []
        return messages
        
    def get_stats(self):
        """Get communication statistics"""
        return {
            'total_messages': len(self.messages),
            'avg_messages_per_step': len(self.messages) / max(1, len(set(m['timestamp'] for m in self.messages))),
            'message_types': self._count_message_types()
        }
        
    def _count_message_types(self):
        types = {}
        for msg in self.messages:
            types[msg['type']] = types.get(msg['type'], 0) + 1
        return types

    def reset(self):
        """Reset for new task"""
        self.step_messages = []