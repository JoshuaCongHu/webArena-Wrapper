
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from WebArenaMetrics import WebArenaMetrics, ActionMetric


@dataclass
class WebArenaTaskMetric:
    """WebArena-specific task tracking"""
    task_id: str
    task_type: str  # 'shopping', 'reddit', 'gitlab', 'map', 'cms'
    site: str
    intent: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    steps_completed: int = 0
    expected_steps: Optional[int] = None
    final_url: Optional[str] = None
    error_message: Optional[str] = None
    screenshot_paths: List[str] = None
    
    def __post_init__(self):
        if self.screenshot_paths is None:
            self.screenshot_paths = []


@dataclass
class WebElementMetric:
    """Web element interaction tracking"""
    timestamp: float
    element_type: str  # 'button', 'input', 'link', 'select', etc.
    element_id: Optional[str] = None
    element_text: Optional[str] = None
    action_type: str = ""  # 'click', 'type', 'select', etc.
    success: bool = False
    xpath: Optional[str] = None
    css_selector: Optional[str] = None
    page_url: str = ""
    agent_id: str = ""


class WebArenaSpecificMetrics(WebArenaMetrics):
    """Enhanced metrics specifically for WebArena tasks with web navigation tracking"""
    
    def __init__(self):
        super().__init__()
        self.task_metrics: List[WebArenaTaskMetric] = []
        self.web_element_metrics: List[WebElementMetric] = []
        self.navigation_history: List[Dict[str, Any]] = []
        self.screenshot_count = 0
        
        # WebArena-specific performance tracking
        self.site_performance: Dict[str, Dict] = {}
        self.task_type_performance: Dict[str, Dict] = {}
        
    def start_task(self, task_id: str, task_type: str, site: str, intent: str, 
                   expected_steps: Optional[int] = None) -> str:
        """Start tracking a new WebArena task"""
        task_metric = WebArenaTaskMetric(
            task_id=task_id,
            task_type=task_type,
            site=site,
            intent=intent,
            start_time=time.time(),
            expected_steps=expected_steps
        )
        self.task_metrics.append(task_metric)
        return task_id
    
    def complete_task(self, task_id: str, success: bool, final_url: Optional[str] = None,
                     error_message: Optional[str] = None) -> None:
        """Mark a task as completed"""
        task = self._find_task(task_id)
        if task:
            task.end_time = time.time()
            task.success = success
            task.final_url = final_url
            task.error_message = error_message
            self._update_site_performance(task)
            self._update_task_type_performance(task)
    
    def track_web_element_interaction(self, agent_id: str, element_type: str, 
                                    action_type: str, success: bool,
                                    element_id: Optional[str] = None,
                                    element_text: Optional[str] = None,
                                    xpath: Optional[str] = None,
                                    css_selector: Optional[str] = None,
                                    page_url: str = "") -> None:
        """Track interaction with web elements"""
        metric = WebElementMetric(
            timestamp=time.time(),
            element_type=element_type,
            element_id=element_id,
            element_text=element_text,
            action_type=action_type,
            success=success,
            xpath=xpath,
            css_selector=css_selector,
            page_url=page_url,
            agent_id=agent_id
        )
        self.web_element_metrics.append(metric)
        
        # Also track as regular action for compatibility
        self.track_action(
            agent_id=agent_id,
            action_type=f"{action_type}_{element_type}",
            success=success,
            execution_time=0.1,  # Assume quick web actions
            cost=0.001  # Minimal cost for web interactions
        )
    
    def track_navigation(self, agent_id: str, from_url: str, to_url: str, 
                        navigation_type: str = "click", success: bool = True) -> None:
        """Track page navigation"""
        nav_metric = {
            'timestamp': time.time(),
            'agent_id': agent_id,
            'from_url': from_url,
            'to_url': to_url,
            'navigation_type': navigation_type,
            'success': success
        }
        self.navigation_history.append(nav_metric)
        
        # Track as action
        self.track_action(
            agent_id=agent_id,
            action_type=f"navigate_{navigation_type}",
            success=success,
            execution_time=0.5,  # Navigation takes a bit longer
            cost=0.002
        )
    
    def track_screenshot(self, agent_id: str, task_id: str, screenshot_path: str,
                        page_url: str = "", description: str = "") -> None:
        """Track screenshot capture"""
        self.screenshot_count += 1
        
        # Add to task screenshots
        task = self._find_task(task_id)
        if task and screenshot_path not in task.screenshot_paths:
            task.screenshot_paths.append(screenshot_path)
        
        # Track as action
        self.track_action(
            agent_id=agent_id,
            action_type="screenshot",
            success=True,
            execution_time=0.2,
            cost=0.001
        )
    
    def increment_task_steps(self, task_id: str) -> None:
        """Increment completed steps for a task"""
        task = self._find_task(task_id)
        if task:
            task.steps_completed += 1
    
    def _find_task(self, task_id: str) -> Optional[WebArenaTaskMetric]:
        """Find task by ID"""
        for task in self.task_metrics:
            if task.task_id == task_id:
                return task
        return None
    
    def _update_site_performance(self, task: WebArenaTaskMetric) -> None:
        """Update performance metrics for a site"""
        site = task.site
        if site not in self.site_performance:
            self.site_performance[site] = {
                'total_tasks': 0,
                'successful_tasks': 0,
                'total_time': 0.0,
                'avg_steps': 0.0,
                'total_steps': 0
            }
        
        perf = self.site_performance[site]
        perf['total_tasks'] += 1
        perf['total_steps'] += task.steps_completed
        
        if task.success:
            perf['successful_tasks'] += 1
            
        if task.end_time:
            perf['total_time'] += (task.end_time - task.start_time)
            
        perf['avg_steps'] = perf['total_steps'] / perf['total_tasks']
    
    def _update_task_type_performance(self, task: WebArenaTaskMetric) -> None:
        """Update performance metrics for a task type"""
        task_type = task.task_type
        if task_type not in self.task_type_performance:
            self.task_type_performance[task_type] = {
                'total_tasks': 0,
                'successful_tasks': 0,
                'total_time': 0.0,
                'avg_steps': 0.0,
                'total_steps': 0
            }
        
        perf = self.task_type_performance[task_type]
        perf['total_tasks'] += 1
        perf['total_steps'] += task.steps_completed
        
        if task.success:
            perf['successful_tasks'] += 1
            
        if task.end_time:
            perf['total_time'] += (task.end_time - task.start_time)
            
        perf['avg_steps'] = perf['total_steps'] / perf['total_tasks']
    
    def get_webarena_summary(self) -> Dict[str, Any]:
        """Get comprehensive WebArena-specific metrics summary"""
        base_summary = self.get_task_summary()
        
        # Task completion metrics
        completed_tasks = [t for t in self.task_metrics if t.end_time is not None]
        successful_tasks = [t for t in completed_tasks if t.success]
        
        task_completion_rate = len(successful_tasks) / len(completed_tasks) if completed_tasks else 0
        avg_task_time = (sum((t.end_time - t.start_time) for t in completed_tasks) / 
                        len(completed_tasks) if completed_tasks else 0)
        avg_steps_per_task = (sum(t.steps_completed for t in completed_tasks) / 
                             len(completed_tasks) if completed_tasks else 0)
        
        # Web element interaction analysis
        element_interactions = len(self.web_element_metrics)
        successful_interactions = sum(1 for e in self.web_element_metrics if e.success)
        interaction_success_rate = successful_interactions / element_interactions if element_interactions else 0
        
        # Navigation analysis
        successful_navigations = sum(1 for n in self.navigation_history if n['success'])
        navigation_success_rate = successful_navigations / len(self.navigation_history) if self.navigation_history else 0
        
        # Element type breakdown
        element_type_breakdown = {}
        for element in self.web_element_metrics:
            elem_type = element.element_type
            if elem_type not in element_type_breakdown:
                element_type_breakdown[elem_type] = {'total': 0, 'successful': 0}
            element_type_breakdown[elem_type]['total'] += 1
            if element.success:
                element_type_breakdown[elem_type]['successful'] += 1
        
        # Add success rates
        for elem_type in element_type_breakdown:
            data = element_type_breakdown[elem_type]
            data['success_rate'] = data['successful'] / data['total'] if data['total'] > 0 else 0
        
        # Site performance analysis
        site_success_rates = {}
        for site, perf in self.site_performance.items():
            site_success_rates[site] = {
                'success_rate': perf['successful_tasks'] / perf['total_tasks'] if perf['total_tasks'] > 0 else 0,
                'avg_time': perf['total_time'] / perf['total_tasks'] if perf['total_tasks'] > 0 else 0,
                'avg_steps': perf['avg_steps'],
                'total_tasks': perf['total_tasks']
            }
        
        # Task type analysis
        task_type_success_rates = {}
        for task_type, perf in self.task_type_performance.items():
            task_type_success_rates[task_type] = {
                'success_rate': perf['successful_tasks'] / perf['total_tasks'] if perf['total_tasks'] > 0 else 0,
                'avg_time': perf['total_time'] / perf['total_tasks'] if perf['total_tasks'] > 0 else 0,
                'avg_steps': perf['avg_steps'],
                'total_tasks': perf['total_tasks']
            }
        
        webarena_metrics = {
            'webarena_task_metrics': {
                'total_tasks': len(self.task_metrics),
                'completed_tasks': len(completed_tasks),
                'successful_tasks': len(successful_tasks),
                'task_completion_rate': task_completion_rate,
                'average_task_duration': avg_task_time,
                'average_steps_per_task': avg_steps_per_task,
                'total_screenshots': self.screenshot_count
            },
            'web_interaction_metrics': {
                'total_element_interactions': element_interactions,
                'successful_interactions': successful_interactions,
                'interaction_success_rate': interaction_success_rate,
                'element_type_breakdown': element_type_breakdown,
                'total_navigations': len(self.navigation_history),
                'successful_navigations': successful_navigations,
                'navigation_success_rate': navigation_success_rate
            },
            'site_performance': site_success_rates,
            'task_type_performance': task_type_success_rates,
            'detailed_tasks': [
                {
                    'task_id': t.task_id,
                    'task_type': t.task_type,
                    'site': t.site,
                    'intent': t.intent,
                    'success': t.success,
                    'steps_completed': t.steps_completed,
                    'expected_steps': t.expected_steps,
                    'duration': (t.end_time - t.start_time) if t.end_time else None,
                    'screenshots': len(t.screenshot_paths),
                    'final_url': t.final_url,
                    'error_message': t.error_message
                }
                for t in self.task_metrics
            ]
        }
        
        # Merge with base summary
        base_summary['webarena_specific'] = webarena_metrics
        return base_summary
    
    def export_webarena_results(self, filepath: str) -> None:
        """Export comprehensive results to JSON file"""
        results = self.get_webarena_summary()
        results['export_timestamp'] = datetime.now().isoformat()
        results['total_session_duration'] = time.time() - self.session_start
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def get_task_efficiency_analysis(self) -> Dict[str, Any]:
        """Analyze task efficiency vs expected performance"""
        analysis = {
            'step_efficiency': {},
            'time_efficiency': {},
            'success_patterns': {},
            'failure_analysis': {}
        }
        
        completed_tasks = [t for t in self.task_metrics if t.end_time is not None]
        
        # Step efficiency analysis
        tasks_with_expected_steps = [t for t in completed_tasks if t.expected_steps is not None]
        if tasks_with_expected_steps:
            step_ratios = []
            for task in tasks_with_expected_steps:
                if task.expected_steps > 0:
                    ratio = task.steps_completed / task.expected_steps
                    step_ratios.append(ratio)
            
            if step_ratios:
                analysis['step_efficiency'] = {
                    'average_step_ratio': sum(step_ratios) / len(step_ratios),
                    'efficient_tasks': sum(1 for r in step_ratios if 0.8 <= r <= 1.2),
                    'over_engineered_tasks': sum(1 for r in step_ratios if r > 1.2),
                    'under_planned_tasks': sum(1 for r in step_ratios if r < 0.8)
                }
        
        # Success patterns by site and task type
        successful_tasks = [t for t in completed_tasks if t.success]
        failed_tasks = [t for t in completed_tasks if not t.success]
        
        if successful_tasks:
            success_sites = {}
            success_types = {}
            
            for task in successful_tasks:
                success_sites[task.site] = success_sites.get(task.site, 0) + 1
                success_types[task.task_type] = success_types.get(task.task_type, 0) + 1
            
            analysis['success_patterns'] = {
                'most_successful_site': max(success_sites.items(), key=lambda x: x[1]) if success_sites else None,
                'most_successful_task_type': max(success_types.items(), key=lambda x: x[1]) if success_types else None,
                'site_breakdown': success_sites,
                'task_type_breakdown': success_types
            }
        
        # Failure analysis
        if failed_tasks:
            failure_reasons = {}
            failure_sites = {}
            failure_types = {}
            
            for task in failed_tasks:
                if task.error_message:
                    reason = task.error_message.split(':')[0]  # Get first part of error
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                
                failure_sites[task.site] = failure_sites.get(task.site, 0) + 1
                failure_types[task.task_type] = failure_types.get(task.task_type, 0) + 1
            
            analysis['failure_analysis'] = {
                'common_failure_reasons': sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True),
                'most_problematic_site': max(failure_sites.items(), key=lambda x: x[1]) if failure_sites else None,
                'most_problematic_task_type': max(failure_types.items(), key=lambda x: x[1]) if failure_types else None,
                'site_failure_breakdown': failure_sites,
                'task_type_failure_breakdown': failure_types
            }
        
        return analysis


class WebNavigationAgent:
    """Agent specialized for web navigation"""
    
    def __init__(self, agent_id: str, model_name: str = "gpt-4"):
        self.agent_id = agent_id
        self.model_name = model_name
        self.cost_tracker = None  # Should be injected
        self.llm = None  # Should be injected
    
    def generate_action(self, state, allowed_actions):
        # Use LLM to understand page and generate action
        prompt = f"""
        Goal: {state['goal']}
        Current page elements: {state['current_page']['elements']}
        Previous actions: {state['action_history'][-3:]}
        
        What's the next action to achieve the goal?
        """
        
        response = self.llm.generate(prompt)
        action = self.parse_action(response)
        
        # Track cost
        if self.cost_tracker:
            self.cost_tracker.track_llm_call(
                model=self.model_name,
                tokens_in=self._count_tokens(prompt),
                tokens_out=self._count_tokens(response)
            )
        
        return action
    
    def parse_action(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into action"""
        # Simple parsing logic - should be enhanced
        return {
            'type': 'click',
            'target': 'button',
            'description': response.strip()
        }
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count - simple approximation"""
        return len(text.split()) * 1.3  # Rough approximation