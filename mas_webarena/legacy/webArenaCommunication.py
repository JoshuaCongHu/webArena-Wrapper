import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class MessageType(Enum):
    """Types of inter-agent communication"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    COORDINATION = "coordination"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    RESOURCE_REQUEST = "resource_request"


@dataclass
class CommunicationMessage:
    """Single communication event between agents"""
    timestamp: float
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    message_size: int
    processing_time: Optional[float] = None


class WebArenaCommunicationMonitor:
    """Monitor and analyze inter-agent communication patterns"""
    
    def __init__(self):
        self.messages: List[CommunicationMessage] = []
        self.agent_stats: Dict[str, Dict] = {}
        self.communication_patterns: Dict[str, int] = {}
        
    def log_message(self, sender_id: str, receiver_id: str, 
                   message_type: MessageType, content: Dict[str, Any],
                   processing_time: Optional[float] = None):
        """Log a communication message between agents"""
        message = CommunicationMessage(
            timestamp=time.time(),
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            message_size=len(str(content)),
            processing_time=processing_time
        )
        
        self.messages.append(message)
        self._update_agent_stats(message)
        self._update_communication_patterns(sender_id, receiver_id)
    
    def _update_agent_stats(self, message: CommunicationMessage):
        """Update communication statistics for agents"""
        # Update sender stats
        if message.sender_id not in self.agent_stats:
            self.agent_stats[message.sender_id] = {
                'messages_sent': 0,
                'messages_received': 0,
                'total_data_sent': 0,
                'total_data_received': 0,
                'avg_response_time': 0.0,
                'message_types_sent': {},
                'message_types_received': {}
            }
        
        sender_stats = self.agent_stats[message.sender_id]
        sender_stats['messages_sent'] += 1
        sender_stats['total_data_sent'] += message.message_size
        
        msg_type = message.message_type.value
        if msg_type not in sender_stats['message_types_sent']:
            sender_stats['message_types_sent'][msg_type] = 0
        sender_stats['message_types_sent'][msg_type] += 1
        
        # Update receiver stats
        if message.receiver_id not in self.agent_stats:
            self.agent_stats[message.receiver_id] = {
                'messages_sent': 0,
                'messages_received': 0,
                'total_data_sent': 0,
                'total_data_received': 0,
                'avg_response_time': 0.0,
                'message_types_sent': {},
                'message_types_received': {}
            }
        
        receiver_stats = self.agent_stats[message.receiver_id]
        receiver_stats['messages_received'] += 1
        receiver_stats['total_data_received'] += message.message_size
        
        if msg_type not in receiver_stats['message_types_received']:
            receiver_stats['message_types_received'][msg_type] = 0
        receiver_stats['message_types_received'][msg_type] += 1
    
    def _update_communication_patterns(self, sender_id: str, receiver_id: str):
        """Track communication patterns between agent pairs"""
        pattern_key = f"{sender_id}->{receiver_id}"
        if pattern_key not in self.communication_patterns:
            self.communication_patterns[pattern_key] = 0
        self.communication_patterns[pattern_key] += 1
    
    def get_communication_efficiency(self) -> Dict[str, Any]:
        """Calculate overall communication efficiency metrics"""
        if not self.messages:
            return {}
        
        total_messages = len(self.messages)
        total_data = sum(msg.message_size for msg in self.messages)
        
        # Calculate average response times for request-response pairs
        response_times = [msg.processing_time for msg in self.messages 
                         if msg.processing_time is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Message type distribution
        type_distribution = {}
        for msg in self.messages:
            msg_type = msg.message_type.value
            type_distribution[msg_type] = type_distribution.get(msg_type, 0) + 1
        
        return {
            'total_messages': total_messages,
            'total_data_transferred': total_data,
            'average_message_size': total_data / total_messages if total_messages > 0 else 0,
            'average_response_time': avg_response_time,
            'message_type_distribution': type_distribution,
            'unique_communication_pairs': len(self.communication_patterns),
            'most_active_communication_pair': max(self.communication_patterns.items(), 
                                                key=lambda x: x[1]) if self.communication_patterns else None
        }
    
    def get_agent_communication_profile(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed communication profile for a specific agent"""
        if agent_id not in self.agent_stats:
            return {}
        
        stats = self.agent_stats[agent_id]
        
        # Calculate communication ratios
        total_messages = stats['messages_sent'] + stats['messages_received']
        send_ratio = stats['messages_sent'] / total_messages if total_messages > 0 else 0
        receive_ratio = stats['messages_received'] / total_messages if total_messages > 0 else 0
        
        # Find communication partners
        partners_sent = set()
        partners_received = set()
        
        for msg in self.messages:
            if msg.sender_id == agent_id:
                partners_sent.add(msg.receiver_id)
            elif msg.receiver_id == agent_id:
                partners_received.add(msg.sender_id)
        
        return {
            **stats,
            'total_communications': total_messages,
            'send_ratio': send_ratio,
            'receive_ratio': receive_ratio,
            'unique_partners_contacted': len(partners_sent),
            'unique_partners_received_from': len(partners_received),
            'avg_message_size_sent': stats['total_data_sent'] / stats['messages_sent'] if stats['messages_sent'] > 0 else 0,
            'avg_message_size_received': stats['total_data_received'] / stats['messages_received'] if stats['messages_received'] > 0 else 0
        }
    
    def get_communication_network_analysis(self) -> Dict[str, Any]:
        """Analyze the overall communication network structure"""
        # Build adjacency information
        agents = set()
        for msg in self.messages:
            agents.add(msg.sender_id)
            agents.add(msg.receiver_id)
        
        # Calculate network metrics
        network_density = len(self.communication_patterns) / (len(agents) * (len(agents) - 1)) if len(agents) > 1 else 0
        
        # Find most central agents (highest communication volume)
        agent_centrality = {}
        for agent in agents:
            sent = sum(1 for msg in self.messages if msg.sender_id == agent)
            received = sum(1 for msg in self.messages if msg.receiver_id == agent)
            agent_centrality[agent] = sent + received
        
        most_central = max(agent_centrality.items(), key=lambda x: x[1]) if agent_centrality else None
        
        # Communication bottlenecks (agents with high receive but low send ratios)
        bottlenecks = []
        for agent in agents:
            profile = self.get_agent_communication_profile(agent)
            if profile and profile.get('receive_ratio', 0) > 0.7 and profile.get('messages_sent', 0) < 5:
                bottlenecks.append(agent)
        
        return {
            'total_agents': len(agents),
            'network_density': network_density,
            'total_communication_pairs': len(self.communication_patterns),
            'most_central_agent': most_central,
            'potential_bottlenecks': bottlenecks,
            'communication_patterns': dict(sorted(self.communication_patterns.items(), 
                                                key=lambda x: x[1], reverse=True))
        }
    
    def detect_communication_anomalies(self) -> List[Dict[str, Any]]:
        """Detect unusual communication patterns that might indicate issues"""
        anomalies = []
        
        # Check for excessive communication (spam detection)
        for agent_id, stats in self.agent_stats.items():
            if stats['messages_sent'] > 100:  # Threshold for excessive messaging
                anomalies.append({
                    'type': 'excessive_messaging',
                    'agent_id': agent_id,
                    'message_count': stats['messages_sent'],
                    'description': f"Agent {agent_id} has sent {stats['messages_sent']} messages"
                })
        
        # Check for communication failures (high error rates)
        error_messages = [msg for msg in self.messages if msg.message_type == MessageType.ERROR_REPORT]
        if len(error_messages) > len(self.messages) * 0.1:  # More than 10% error rate
            anomalies.append({
                'type': 'high_error_rate',
                'error_count': len(error_messages),
                'total_messages': len(self.messages),
                'error_rate': len(error_messages) / len(self.messages),
                'description': f"High error rate: {len(error_messages)}/{len(self.messages)} messages are errors"
            })
        
        # Check for isolated agents (no communication)
        all_agents = set(self.agent_stats.keys())
        communicating_agents = set()
        for msg in self.messages:
            communicating_agents.add(msg.sender_id)
            communicating_agents.add(msg.receiver_id)
        
        isolated_agents = all_agents - communicating_agents
        for agent in isolated_agents:
            anomalies.append({
                'type': 'isolated_agent',
                'agent_id': agent,
                'description': f"Agent {agent} has no recorded communications"
            })
        
        return anomalies