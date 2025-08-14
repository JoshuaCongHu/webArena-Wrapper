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
    # LLM orchestrator specific fields
    node_id: Optional[int] = None
    dag_confidence: Optional[float] = None
    replanned: bool = False
    llm_tokens_used: Optional[int] = None


@dataclass
class DAGMetric:
    """DAG generation and execution tracking"""
    timestamp: float
    dag_id: str
    generation_method: str  # 'llm' or 'fallback'
    llm_model: Optional[str]
    node_count: int
    edge_count: int
    confidence_score: float
    generation_time: float
    generation_cost: float
    validation_passed: bool
    cache_hit: bool = False
    replanning_trigger: Optional[str] = None


@dataclass
class ReplanningMetric:
    """Replanning event tracking"""
    timestamp: float
    original_dag_id: str
    new_dag_id: str
    trigger_reason: str
    completed_nodes: List[int]
    remaining_budget: float
    replan_generation_time: float
    replan_cost: float


class WebArenaMetrics:
    """Enhanced metrics collection for LLM-based WebArena MAS"""
    
    def __init__(self):
        self.session_start = time.time()
        self.actions: List[ActionMetric] = []
        self.agent_performance: Dict[str, Dict] = {}
        self.task_metrics: Dict[str, Any] = {}
        
        # LLM orchestrator specific tracking
        self.dag_metrics: List[DAGMetric] = []
        self.replanning_metrics: List[ReplanningMetric] = []
        self.llm_usage: Dict[str, Any] = {
            'total_calls': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'generation_failures': 0,
            'fallback_generations': 0
        }
        
    def track_action(self, agent_id: str, action_type: str, success: bool, 
                    execution_time: float, cost: float = 0.0, 
                    error_message: Optional[str] = None, node_id: Optional[int] = None,
                    dag_confidence: Optional[float] = None, replanned: bool = False,
                    llm_tokens_used: Optional[int] = None):
        """Track a single agent action with LLM orchestrator context"""
        metric = ActionMetric(
            timestamp=time.time(),
            action_type=action_type,
            agent_id=agent_id,
            success=success,
            execution_time=execution_time,
            cost=cost,
            error_message=error_message,
            node_id=node_id,
            dag_confidence=dag_confidence,
            replanned=replanned,
            llm_tokens_used=llm_tokens_used
        )
        self.actions.append(metric)
        self._update_agent_performance(agent_id, metric)
        
        # Update LLM usage if tokens were used
        if llm_tokens_used:
            self.llm_usage['total_tokens'] += llm_tokens_used
    
    def track_dag_generation(self, dag_id: str, generation_method: str, 
                           llm_model: Optional[str], node_count: int, edge_count: int,
                           confidence_score: float, generation_time: float,
                           generation_cost: float, validation_passed: bool,
                           cache_hit: bool = False, replanning_trigger: Optional[str] = None):
        """Track DAG generation event"""
        metric = DAGMetric(
            timestamp=time.time(),
            dag_id=dag_id,
            generation_method=generation_method,
            llm_model=llm_model,
            node_count=node_count,
            edge_count=edge_count,
            confidence_score=confidence_score,
            generation_time=generation_time,
            generation_cost=generation_cost,
            validation_passed=validation_passed,
            cache_hit=cache_hit,
            replanning_trigger=replanning_trigger
        )
        self.dag_metrics.append(metric)
        
        # Update LLM usage statistics
        if generation_method == 'llm':
            self.llm_usage['total_calls'] += 1
            self.llm_usage['total_cost'] += generation_cost
            if not validation_passed:
                self.llm_usage['generation_failures'] += 1
        elif generation_method == 'fallback':
            self.llm_usage['fallback_generations'] += 1
            
        if cache_hit:
            self.llm_usage['cache_hits'] += 1
        else:
            self.llm_usage['cache_misses'] += 1
    
    def track_replanning(self, original_dag_id: str, new_dag_id: str, 
                        trigger_reason: str, completed_nodes: List[int],
                        remaining_budget: float, replan_generation_time: float,
                        replan_cost: float):
        """Track replanning event"""
        metric = ReplanningMetric(
            timestamp=time.time(),
            original_dag_id=original_dag_id,
            new_dag_id=new_dag_id,
            trigger_reason=trigger_reason,
            completed_nodes=completed_nodes,
            remaining_budget=remaining_budget,
            replan_generation_time=replan_generation_time,
            replan_cost=replan_cost
        )
        self.replanning_metrics.append(metric)
        self.llm_usage['total_cost'] += replan_cost
    
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
        """Get comprehensive task execution summary including LLM orchestrator metrics"""
        efficiency = self.get_efficiency_metrics()
        agent_summaries = {
            agent_id: self.get_agent_performance(agent_id) 
            for agent_id in self.agent_performance.keys()
        }
        llm_metrics = self.get_llm_orchestrator_metrics()
        cost_analysis = self.get_cost_analysis()
        
        return {
            'session_info': {
                'start_time': datetime.fromtimestamp(self.session_start).isoformat(),
                'duration_seconds': time.time() - self.session_start,
                'llm_orchestrator_enabled': len(self.dag_metrics) > 0
            },
            'efficiency_metrics': efficiency,
            'agent_performance': agent_summaries,
            'action_breakdown': self._get_action_breakdown(),
            'llm_orchestrator_metrics': llm_metrics,
            'cost_analysis': cost_analysis,
            'summary_insights': self._generate_insights()
        }
    
    def _generate_insights(self) -> Dict[str, Any]:
        """Generate actionable insights from metrics"""
        insights = {
            'recommendations': [],
            'warnings': [],
            'performance_highlights': []
        }
        
        if not self.dag_metrics:
            return insights
        
        # Analyze LLM orchestrator performance
        llm_metrics = self.get_llm_orchestrator_metrics()
        
        # Cache hit rate insights
        cache_hit_rate = llm_metrics['llm_usage']['cache_hit_rate']
        if cache_hit_rate > 0.8:
            insights['performance_highlights'].append(f"Excellent cache hit rate: {cache_hit_rate:.1%}")
        elif cache_hit_rate < 0.3:
            insights['recommendations'].append("Consider improving DAG caching strategy - low hit rate")
        
        # Validation failure insights
        validation_failure_rate = llm_metrics['dag_generation']['validation_failure_rate']
        if validation_failure_rate > 0.2:
            insights['warnings'].append(f"High DAG validation failure rate: {validation_failure_rate:.1%}")
            insights['recommendations'].append("Review LLM prompt engineering - frequent validation failures")
        
        # Replanning insights
        replanning_rate = llm_metrics['replanning']['replanning_rate']
        if replanning_rate > 0.5:
            insights['warnings'].append(f"High replanning rate: {replanning_rate:.1%}")
            insights['recommendations'].append("Investigate common replanning triggers")
        elif replanning_rate == 0:
            insights['performance_highlights'].append("No replanning needed - excellent initial planning")
        
        # Cost insights
        cost_analysis = self.get_cost_analysis()
        llm_cost_pct = cost_analysis['cost_percentages']['llm_orchestration_pct']
        if llm_cost_pct > 50:
            insights['warnings'].append(f"LLM orchestration is {llm_cost_pct:.1f}% of total cost")
            insights['recommendations'].append("Consider optimizing LLM usage or using cheaper models")
        
        # Confidence insights
        avg_confidence = llm_metrics['dag_generation']['average_confidence_score']
        if avg_confidence > 0.8:
            insights['performance_highlights'].append(f"High DAG confidence: {avg_confidence:.2f}")
        elif avg_confidence < 0.6:
            insights['recommendations'].append("DAG confidence is low - review task complexity or model choice")
        
        # Fallback rate insights
        fallback_rate = llm_metrics['llm_usage']['fallback_rate']
        if fallback_rate > 0.3:
            insights['warnings'].append(f"High fallback generation rate: {fallback_rate:.1%}")
            insights['recommendations'].append("Check LLM availability and API key configuration")
        
        return insights
    
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
    
    def get_llm_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get comprehensive LLM orchestrator performance metrics"""
        if not self.dag_metrics:
            return {'message': 'No LLM orchestrator metrics available'}
        
        # DAG generation metrics
        total_dags = len(self.dag_metrics)
        llm_generated = sum(1 for d in self.dag_metrics if d.generation_method == 'llm')
        fallback_generated = sum(1 for d in self.dag_metrics if d.generation_method == 'fallback')
        cache_hits = sum(1 for d in self.dag_metrics if d.cache_hit)
        validation_failures = sum(1 for d in self.dag_metrics if not d.validation_passed)
        
        avg_confidence = sum(d.confidence_score for d in self.dag_metrics) / total_dags
        avg_generation_time = sum(d.generation_time for d in self.dag_metrics) / total_dags
        avg_dag_size = sum(d.node_count for d in self.dag_metrics) / total_dags
        total_dag_cost = sum(d.generation_cost for d in self.dag_metrics)
        
        # Replanning metrics
        total_replans = len(self.replanning_metrics)
        replan_triggers = {}
        for replan in self.replanning_metrics:
            trigger = replan.trigger_reason
            replan_triggers[trigger] = replan_triggers.get(trigger, 0) + 1
        
        avg_replan_time = (sum(r.replan_generation_time for r in self.replanning_metrics) / 
                          total_replans if total_replans > 0 else 0)
        total_replan_cost = sum(r.replan_cost for r in self.replanning_metrics)
        
        # Node execution analysis
        replanned_actions = sum(1 for a in self.actions if a.replanned)
        dag_confidence_actions = [a for a in self.actions if a.dag_confidence is not None]
        avg_action_confidence = (sum(a.dag_confidence for a in dag_confidence_actions) / 
                               len(dag_confidence_actions) if dag_confidence_actions else 0)
        
        return {
            'dag_generation': {
                'total_dags_generated': total_dags,
                'llm_generated': llm_generated,
                'fallback_generated': fallback_generated,
                'cache_hit_rate': cache_hits / total_dags if total_dags > 0 else 0,
                'validation_failure_rate': validation_failures / total_dags if total_dags > 0 else 0,
                'average_confidence_score': avg_confidence,
                'average_generation_time': avg_generation_time,
                'average_dag_size': avg_dag_size,
                'total_generation_cost': total_dag_cost,
                'fallback_rate': fallback_generated / total_dags if total_dags > 0 else 0
            },
            'replanning': {
                'total_replanning_events': total_replans,
                'replanning_triggers': replan_triggers,
                'average_replan_time': avg_replan_time,
                'total_replan_cost': total_replan_cost,
                'replanning_rate': total_replans / total_dags if total_dags > 0 else 0
            },
            'llm_usage': {
                **self.llm_usage,
                'cache_hit_rate': (self.llm_usage['cache_hits'] / 
                                 (self.llm_usage['cache_hits'] + self.llm_usage['cache_misses']) 
                                 if (self.llm_usage['cache_hits'] + self.llm_usage['cache_misses']) > 0 else 0),
                'failure_rate': (self.llm_usage['generation_failures'] / 
                               self.llm_usage['total_calls'] if self.llm_usage['total_calls'] > 0 else 0),
                'fallback_rate': (self.llm_usage['fallback_generations'] / 
                                (self.llm_usage['total_calls'] + self.llm_usage['fallback_generations'])
                                if (self.llm_usage['total_calls'] + self.llm_usage['fallback_generations']) > 0 else 0)
            },
            'execution': {
                'replanned_actions': replanned_actions,
                'replanned_action_rate': replanned_actions / len(self.actions) if self.actions else 0,
                'average_action_confidence': avg_action_confidence,
                'high_confidence_actions': sum(1 for a in dag_confidence_actions if a.dag_confidence > 0.8),
                'low_confidence_actions': sum(1 for a in dag_confidence_actions if a.dag_confidence < 0.5)
            }
        }
    
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Detailed cost breakdown including LLM usage"""
        base_metrics = self.get_efficiency_metrics()
        
        # LLM-specific costs
        llm_orchestrator_cost = sum(d.generation_cost for d in self.dag_metrics)
        replanning_cost = sum(r.replan_cost for r in self.replanning_metrics)
        action_execution_cost = sum(a.cost for a in self.actions)
        
        total_system_cost = llm_orchestrator_cost + replanning_cost + action_execution_cost
        
        return {
            'total_system_cost': total_system_cost,
            'cost_breakdown': {
                'llm_orchestration': llm_orchestrator_cost,
                'replanning': replanning_cost,
                'action_execution': action_execution_cost
            },
            'cost_percentages': {
                'llm_orchestration_pct': (llm_orchestrator_cost / total_system_cost * 100 
                                        if total_system_cost > 0 else 0),
                'replanning_pct': (replanning_cost / total_system_cost * 100 
                                 if total_system_cost > 0 else 0),
                'action_execution_pct': (action_execution_cost / total_system_cost * 100 
                                       if total_system_cost > 0 else 0)
            },
            'efficiency_metrics': {
                'cost_per_dag': (llm_orchestrator_cost / len(self.dag_metrics) 
                               if self.dag_metrics else 0),
                'cost_per_replan': (replanning_cost / len(self.replanning_metrics) 
                                  if self.replanning_metrics else 0),
                'cost_per_action': base_metrics.get('cost_per_action', 0)
            }
        }