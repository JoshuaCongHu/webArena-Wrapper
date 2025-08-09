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
        return len(text.split()) * 1.3  # Rough approximation ask others for opionion 