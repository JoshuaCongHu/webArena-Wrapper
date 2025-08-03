import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ActionMetric:
    """Single action tracking metric"""
    timestamp: float
    action_type: str
    agent_id: str
    success: bool
    execution_time: float
    cost: float
    error_message: Optional[str] = None


class WebArenaMetrics:
    """Core metrics collection and analysis for WebArena MAS"""
    
    def __init__(self):
        self.session_start = time.time()
        self.actions: List[ActionMetric] = []
        self.agent_performance: Dict[str, Dict] = {}
        self.task_metrics: Dict[str, Any] = {}
        
    def track_action(self, agent_id: str, action_type: str, success: bool, 
                    execution_time: float, cost: float = 0.0, 
                    error_message: Optional[str] = None):
        """Track a single agent action"""
        metric = ActionMetric(
            timestamp=time.time(),
            action_type=action_type,
            agent_id=agent_id,
            success=success,
            execution_time=execution_time,
            cost=cost,
            error_message=error_message
        )
        self.actions.append(metric)
        self._update_agent_performance(agent_id, metric)
    
    def _update_agent_performance(self, agent_id: str, metric: ActionMetric):
        """Update running performance stats for an agent"""
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                'total_actions': 0,
                'successful_actions': 0,
                'total_cost': 0.0,
                'total_time': 0.0,
                'action_types': {}
            }
        
        perf = self.agent_performance[agent_id]
        perf['total_actions'] += 1
        perf['total_cost'] += metric.cost
        perf['total_time'] += metric.execution_time
        
        if metric.success:
            perf['successful_actions'] += 1
            
        if metric.action_type not in perf['action_types']:
            perf['action_types'][metric.action_type] = {'count': 0, 'success_rate': 0.0}
        perf['action_types'][metric.action_type]['count'] += 1
    
    def get_efficiency_metrics(self) -> Dict[str, Any]:
        """Calculate overall system efficiency metrics"""
        if not self.actions:
            return {}
            
        total_actions = len(self.actions)
        successful_actions = sum(1 for a in self.actions if a.success)
        total_time = sum(a.execution_time for a in self.actions)
        total_cost = sum(a.cost for a in self.actions)
        
        return {
            'total_actions': total_actions,
            'success_rate': successful_actions / total_actions if total_actions > 0 else 0,
            'average_execution_time': total_time / total_actions if total_actions > 0 else 0,
            'total_cost': total_cost,
            'cost_per_action': total_cost / total_actions if total_actions > 0 else 0,
            'session_duration': time.time() - self.session_start,
            'actions_per_minute': total_actions / ((time.time() - self.session_start) / 60) if time.time() - self.session_start > 0 else 0
        }
    
    def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific agent"""
        if agent_id not in self.agent_performance:
            return {}
            
        perf = self.agent_performance[agent_id]
        agent_actions = [a for a in self.actions if a.agent_id == agent_id]
        
        if not agent_actions:
            return perf
            
        success_rate = perf['successful_actions'] / perf['total_actions'] if perf['total_actions'] > 0 else 0
        avg_time = perf['total_time'] / perf['total_actions'] if perf['total_actions'] > 0 else 0
        
        # Calculate action type success rates
        for action_type in perf['action_types']:
            type_actions = [a for a in agent_actions if a.action_type == action_type]
            successful_type_actions = sum(1 for a in type_actions if a.success)
            perf['action_types'][action_type]['success_rate'] = (
                successful_type_actions / len(type_actions) if type_actions else 0
            )
        
        return {
            **perf,
            'success_rate': success_rate,
            'average_execution_time': avg_time,
            'cost_per_action': perf['total_cost'] / perf['total_actions'] if perf['total_actions'] > 0 else 0
        }
    
    def get_task_summary(self) -> Dict[str, Any]:
        """Get comprehensive task execution summary"""
        efficiency = self.get_efficiency_metrics()
        agent_summaries = {
            agent_id: self.get_agent_performance(agent_id) 
            for agent_id in self.agent_performance.keys()
        }
        
        return {
            'session_info': {
                'start_time': datetime.fromtimestamp(self.session_start).isoformat(),
                'duration_seconds': time.time() - self.session_start
            },
            'efficiency_metrics': efficiency,
            'agent_performance': agent_summaries,
            'action_breakdown': self._get_action_breakdown()
        }
    
    def _get_action_breakdown(self) -> Dict[str, Any]:
        """Break down actions by type and success/failure"""
        breakdown = {}
        for action in self.actions:
            action_type = action.action_type
            if action_type not in breakdown:
                breakdown[action_type] = {
                    'total': 0,
                    'successful': 0,
                    'failed': 0,
                    'total_time': 0.0,
                    'total_cost': 0.0
                }
            
            breakdown[action_type]['total'] += 1
            breakdown[action_type]['total_time'] += action.execution_time
            breakdown[action_type]['total_cost'] += action.cost
            
            if action.success:
                breakdown[action_type]['successful'] += 1
            else:
                breakdown[action_type]['failed'] += 1
        
        # Calculate success rates and averages
        for action_type in breakdown:
            data = breakdown[action_type]
            data['success_rate'] = data['successful'] / data['total'] if data['total'] > 0 else 0
            data['avg_time'] = data['total_time'] / data['total'] if data['total'] > 0 else 0
            data['avg_cost'] = data['total_cost'] / data['total'] if data['total'] > 0 else 0
            
        return breakdown