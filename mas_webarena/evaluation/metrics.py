"""
Research metrics calculation and evaluation system for WebArena MAS experiments
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import scipy.stats as stats
from collections import defaultdict
import math


class ResearchMetrics:
    """Calculate all research metrics for MAS WebArena experiments"""
    
    @staticmethod
    def calculate_emergence_metrics(trajectories: List[Dict]) -> Dict[str, float]:
        """Quantify emergent coordination behaviors"""
        
        if not trajectories:
            return {'communication_entropy': 0, 'agent_diversity': 0, 'parallelization_rate': 0}
        
        # Communication entropy (based on agent interactions)
        comm_entropy = ResearchMetrics._calculate_communication_entropy(trajectories)
        
        # Agent diversity (Simpson's diversity index)
        agent_usage = ResearchMetrics._count_agent_usage(trajectories)
        diversity = ResearchMetrics._simpson_diversity(agent_usage)
        
        # Parallelization efficiency
        parallel_steps = ResearchMetrics._count_parallel_steps(trajectories)
        total_steps = len(trajectories)
        parallel_rate = parallel_steps / total_steps if total_steps > 0 else 0
        
        return {
            'communication_entropy': comm_entropy,
            'agent_diversity': diversity,
            'parallelization_rate': parallel_rate
        }
    
    @staticmethod
    def _calculate_communication_entropy(trajectories: List[Dict]) -> float:
        """Calculate entropy of agent communication patterns"""
        
        # Extract communication patterns (agent transitions)
        transitions = []
        prev_agent = None
        
        for step in trajectories:
            current_agent = step.get('agent_id', 0)
            if prev_agent is not None:
                transitions.append((prev_agent, current_agent))
            prev_agent = current_agent
        
        if not transitions:
            return 0.0
        
        # Count transition frequencies
        transition_counts = defaultdict(int)
        for transition in transitions:
            transition_counts[transition] += 1
        
        # Calculate entropy
        total_transitions = len(transitions)
        entropy = 0.0
        
        for count in transition_counts.values():
            prob = count / total_transitions
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    @staticmethod
    def _count_agent_usage(trajectories: List[Dict]) -> Dict[int, int]:
        """Count how often each agent is used"""
        agent_counts = defaultdict(int)
        
        for step in trajectories:
            agent_id = step.get('agent_id', 0)
            agent_counts[agent_id] += 1
        
        return dict(agent_counts)
    
    @staticmethod
    def _simpson_diversity(counts: Dict[int, int]) -> float:
        """Calculate Simpson's diversity index"""
        if not counts:
            return 0.0
        
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        simpson_index = sum((n * (n - 1)) for n in counts.values()) / (total * (total - 1)) if total > 1 else 0
        return 1 - simpson_index  # Simpson's diversity (1 - Simpson's index)
    
    @staticmethod
    def _count_parallel_steps(trajectories: List[Dict]) -> int:
        """Count steps that were executed in parallel"""
        parallel_count = 0
        
        # Group by execution level/timestamp
        level_groups = defaultdict(list)
        for step in trajectories:
            level = step.get('level', 0)
            level_groups[level].append(step)
        
        # Count levels with multiple simultaneous actions
        for level_steps in level_groups.values():
            if len(level_steps) > 1:
                parallel_count += len(level_steps)
        
        return parallel_count
    
    @staticmethod
    def calculate_graph_metrics(dags: List[nx.DiGraph]) -> Dict[str, float]:
        """Calculate DAG complexity and structure metrics"""
        
        if not dags:
            return {
                'avg_nodes': 0, 'avg_edges': 0, 'avg_diameter': 0, 
                'avg_clustering': 0, 'avg_path_length': 0, 'complexity_score': 0
            }
        
        metrics = {
            'avg_nodes': np.mean([g.number_of_nodes() for g in dags if g.number_of_nodes() > 0]),
            'avg_edges': np.mean([g.number_of_edges() for g in dags if g.number_of_nodes() > 0]),
        }
        
        # More complex metrics that require connected graphs
        diameters = []
        clusterings = []
        path_lengths = []
        complexity_scores = []
        
        for g in dags:
            if g.number_of_nodes() == 0:
                continue
                
            # Diameter (longest shortest path)
            if nx.is_weakly_connected(g):
                diameter = nx.diameter(g.to_undirected())
                diameters.append(diameter)
                
                # Average path length
                try:
                    avg_path = nx.average_shortest_path_length(g.to_undirected())
                    path_lengths.append(avg_path)
                except nx.NetworkXError:
                    path_lengths.append(0)
            else:
                diameters.append(-1)  # Disconnected
                path_lengths.append(-1)
            
            # Clustering coefficient
            clustering = nx.average_clustering(g.to_undirected()) if g.number_of_nodes() > 1 else 0
            clusterings.append(clustering)
            
            # Complexity score (combination of structural properties)
            complexity = ResearchMetrics._calculate_dag_complexity(g)
            complexity_scores.append(complexity)
        
        metrics.update({
            'avg_diameter': np.mean(diameters) if diameters else 0,
            'avg_clustering': np.mean(clusterings) if clusterings else 0,
            'avg_path_length': np.mean(path_lengths) if path_lengths else 0,
            'complexity_score': np.mean(complexity_scores) if complexity_scores else 0
        })
        
        return metrics
    
    @staticmethod
    def _calculate_dag_complexity(dag: nx.DiGraph) -> float:
        """Calculate a complexity score for a DAG"""
        if dag.number_of_nodes() == 0:
            return 0.0
        
        n_nodes = dag.number_of_nodes()
        n_edges = dag.number_of_edges()
        
        # Basic complexity: edge density
        max_edges = n_nodes * (n_nodes - 1) / 2  # For DAG
        edge_density = n_edges / max_edges if max_edges > 0 else 0
        
        # Add branching factor
        out_degrees = [dag.out_degree(node) for node in dag.nodes()]
        avg_branching = np.mean(out_degrees) if out_degrees else 0
        
        # Add depth (longest path)
        try:
            depth = nx.dag_longest_path_length(dag) if nx.is_directed_acyclic_graph(dag) else 0
        except:
            depth = 0
        
        # Combine metrics (normalized)
        complexity = (edge_density + avg_branching/n_nodes + depth/n_nodes) / 3
        
        return complexity
    
    @staticmethod
    def calculate_constraint_metrics(costs: List[float], 
                                   budget: float,
                                   alpha: float = 1.05) -> Dict[str, float]:
        """Calculate constraint satisfaction metrics"""
        
        if not costs:
            return {
                'cost_guarantee_rate': 0, 'avg_violation': 0, 'max_violation': 0,
                'violation_severity': 0, 'budget_utilization': 0
            }
        
        costs = np.array(costs)
        budget_limit = budget * alpha
        
        # Cost guarantee rate (% of episodes within budget)
        within_budget = costs <= budget_limit
        cost_guarantee_rate = np.mean(within_budget)
        
        # Violation metrics
        violations = np.maximum(0, costs - budget_limit)
        avg_violation = np.mean(violations)
        max_violation = np.max(violations)
        
        # Violation severity (weighted by magnitude)
        violation_severity = np.mean(violations ** 2) ** 0.5  # RMS violation
        
        # Budget utilization
        budget_utilization = np.mean(costs) / budget if budget > 0 else 0
        
        return {
            'cost_guarantee_rate': cost_guarantee_rate,
            'avg_violation': avg_violation,
            'max_violation': max_violation,
            'violation_severity': violation_severity,
            'budget_utilization': budget_utilization
        }
    
    @staticmethod
    def calculate_learning_metrics(success_rates: List[float],
                                 costs: List[float],
                                 window_size: int = 100) -> Dict[str, float]:
        """Calculate learning progress metrics"""
        
        if len(success_rates) < window_size:
            return {'learning_rate': 0, 'convergence_episode': -1, 'final_performance': 0}
        
        # Learning rate (improvement over time)
        early_performance = np.mean(success_rates[:window_size])
        late_performance = np.mean(success_rates[-window_size:])
        learning_rate = (late_performance - early_performance) / len(success_rates)
        
        # Convergence detection (when performance stabilizes)
        convergence_episode = ResearchMetrics._detect_convergence(success_rates, window_size)
        
        # Final performance
        final_performance = np.mean(success_rates[-window_size//2:])
        
        return {
            'learning_rate': learning_rate,
            'convergence_episode': convergence_episode,
            'final_performance': final_performance
        }
    
    @staticmethod
    def _detect_convergence(values: List[float], window_size: int, 
                           threshold: float = 0.01) -> int:
        """Detect when learning has converged (performance stops improving)"""
        
        if len(values) < 2 * window_size:
            return -1
        
        for i in range(window_size, len(values) - window_size):
            before = np.mean(values[i-window_size:i])
            after = np.mean(values[i:i+window_size])
            
            if abs(after - before) < threshold:
                return i
        
        return -1  # Not converged
    
    @staticmethod
    def calculate_efficiency_metrics(trajectories: List[Dict],
                                   success_rates: List[float]) -> Dict[str, float]:
        """Calculate efficiency metrics"""
        
        if not trajectories:
            return {'avg_episode_length': 0, 'success_per_step': 0, 'action_efficiency': 0}
        
        # Episode length
        episode_lengths = [len(traj) if isinstance(traj, list) else 1 for traj in trajectories]
        avg_episode_length = np.mean(episode_lengths)
        
        # Success per step
        total_steps = sum(episode_lengths)
        total_successes = sum(success_rates) if success_rates else 0
        success_per_step = total_successes / total_steps if total_steps > 0 else 0
        
        # Action efficiency (successful actions / total actions)
        if isinstance(trajectories[0], list):
            successful_actions = sum(
                sum(1 for step in traj if step.get('success', False))
                for traj in trajectories if isinstance(traj, list)
            )
            total_actions = sum(
                len(traj) for traj in trajectories if isinstance(traj, list)
            )
        else:
            successful_actions = sum(step.get('success', False) for step in trajectories)
            total_actions = len(trajectories)
        
        action_efficiency = successful_actions / total_actions if total_actions > 0 else 0
        
        return {
            'avg_episode_length': avg_episode_length,
            'success_per_step': success_per_step,
            'action_efficiency': action_efficiency
        }


class StatisticalAnalysis:
    """Statistical analysis tools for experiment results"""
    
    @staticmethod
    def compare_methods(results_dict: Dict[str, List[float]], 
                       metric_name: str = "performance") -> Dict[str, Any]:
        """Compare multiple methods statistically"""
        
        methods = list(results_dict.keys())
        if len(methods) < 2:
            return {'error': 'Need at least 2 methods to compare'}
        
        # Pairwise comparisons
        comparisons = {}
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                data1 = np.array(results_dict[method1])
                data2 = np.array(results_dict[method2])
                
                # T-test
                t_stat, t_pvalue = stats.ttest_ind(data1, data2)
                
                # Mann-Whitney U test (non-parametric)
                u_stat, u_pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1) + 
                                     (len(data2) - 1) * np.var(data2)) / 
                                    (len(data1) + len(data2) - 2))
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
                
                comparisons[f"{method1}_vs_{method2}"] = {
                    'mean_diff': np.mean(data1) - np.mean(data2),
                    't_statistic': t_stat,
                    't_pvalue': t_pvalue,
                    'u_statistic': u_stat,
                    'u_pvalue': u_pvalue,
                    'cohens_d': cohens_d,
                    'significant_t': t_pvalue < 0.05,
                    'significant_u': u_pvalue < 0.05
                }
        
        # ANOVA for overall comparison
        if len(methods) > 2:
            f_stat, f_pvalue = stats.f_oneway(*[results_dict[method] for method in methods])
            
            anova_result = {
                'f_statistic': f_stat,
                'f_pvalue': f_pvalue,
                'significant': f_pvalue < 0.05
            }
        else:
            anova_result = None
        
        return {
            'metric': metric_name,
            'methods': methods,
            'pairwise_comparisons': comparisons,
            'anova': anova_result,
            'summary': StatisticalAnalysis._create_comparison_summary(results_dict, comparisons)
        }
    
    @staticmethod
    def _create_comparison_summary(results_dict: Dict[str, List[float]], 
                                  comparisons: Dict[str, Dict]) -> Dict[str, Any]:
        """Create a summary of method performance"""
        
        summary = {}
        
        for method, values in results_dict.items():
            summary[method] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'ci_95': stats.t.interval(0.95, len(values)-1, 
                                        loc=np.mean(values), 
                                        scale=stats.sem(values)) if len(values) > 1 else (0, 0)
            }
        
        # Rank methods by performance
        ranked_methods = sorted(summary.keys(), 
                              key=lambda m: summary[m]['mean'], 
                              reverse=True)
        
        summary['ranking'] = ranked_methods
        
        return summary
    
    @staticmethod
    def bootstrap_confidence_interval(data: List[float], 
                                    n_bootstrap: int = 1000, 
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        
        data = np.array(data)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return lower, upper


class PerformanceAnalyzer:
    """Analyze performance patterns and trends"""
    
    @staticmethod
    def analyze_learning_curves(history: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze learning curve characteristics"""
        
        if not history:
            return {'error': 'No history data provided'}
        
        # Extract metrics over time
        episodes = [h.get('episode', i) for i, h in enumerate(history)]
        success_rates = [h.get('success_rate', 0) for h in history]
        costs = [h.get('avg_cost', 0) for h in history]
        
        analysis = {
            'total_episodes': len(episodes),
            'final_success_rate': success_rates[-1] if success_rates else 0,
            'final_cost': costs[-1] if costs else 0,
        }
        
        # Learning trends
        if len(success_rates) >= 10:
            # Linear regression on success rate
            x = np.array(range(len(success_rates)))
            success_slope, success_intercept, success_r_value, _, _ = stats.linregress(x, success_rates)
            
            cost_slope, cost_intercept, cost_r_value, _, _ = stats.linregress(x, costs)
            
            analysis.update({
                'success_trend_slope': success_slope,
                'success_trend_r2': success_r_value ** 2,
                'cost_trend_slope': cost_slope,
                'cost_trend_r2': cost_r_value ** 2,
                'improving_success': success_slope > 0,
                'decreasing_cost': cost_slope < 0
            })
        
        return analysis
    
    @staticmethod
    def identify_performance_regimes(values: List[float], 
                                   window_size: int = 50) -> List[Dict[str, Any]]:
        """Identify different performance regimes in learning"""
        
        if len(values) < window_size * 2:
            return []
        
        regimes = []
        current_regime = {'start': 0, 'values': []}
        
        for i in range(window_size, len(values), window_size):
            window_mean = np.mean(values[i-window_size:i])
            
            if not current_regime['values']:
                current_regime['values'].append(window_mean)
                current_regime['mean'] = window_mean
            else:
                prev_mean = current_regime['mean']
                change_rate = (window_mean - prev_mean) / prev_mean if prev_mean != 0 else 0
                
                # If significant change, start new regime
                if abs(change_rate) > 0.1:  # 10% change threshold
                    current_regime['end'] = i - window_size
                    current_regime['length'] = current_regime['end'] - current_regime['start']
                    regimes.append(current_regime)
                    
                    current_regime = {
                        'start': i - window_size,
                        'values': [window_mean],
                        'mean': window_mean
                    }
                else:
                    current_regime['values'].append(window_mean)
                    current_regime['mean'] = np.mean(current_regime['values'])
        
        # Add final regime
        if current_regime['values']:
            current_regime['end'] = len(values)
            current_regime['length'] = current_regime['end'] - current_regime['start']
            regimes.append(current_regime)
        
        return regimes