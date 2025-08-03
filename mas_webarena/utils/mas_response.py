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