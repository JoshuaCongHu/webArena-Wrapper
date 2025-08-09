"""
Figure generation for research paper visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path
import sys
import os

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import ResearchMetrics, StatisticalAnalysis


class FigureGenerator:
    """Generate all figures for the research paper"""
    
    def __init__(self, results_file: str, output_dir: str = "figures"):
        """Initialize with results file"""
        self.results_file = results_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load results
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        # Set up plotting parameters
        self.figsize = (12, 8)
        self.dpi = 300
        self.method_colors = {
            'p3o': '#E74C3C',           # Red
            'ppo_lagrangian': '#3498DB', # Blue  
            'macpo': '#2ECC71',         # Green
            'single_llm': '#F39C12',    # Orange
            'heuristic_mas': '#9B59B6'  # Purple
        }
        
    def generate_all_figures(self) -> List[Path]:
        """Generate all paper figures"""
        
        print("Generating research figures...")
        generated_files = []
        
        try:
            # Figure 1: Learning curves comparison
            fig1_path = self.plot_learning_curves()
            if fig1_path:
                generated_files.append(fig1_path)
            
            # Figure 2: Pareto frontier (Success vs Cost)
            fig2_path = self.plot_pareto_frontier()
            if fig2_path:
                generated_files.append(fig2_path)
            
            # Figure 3: Cost guarantee satisfaction over time
            fig3_path = self.plot_cost_guarantees()
            if fig3_path:
                generated_files.append(fig3_path)
            
            # Figure 4: DAG complexity evolution
            fig4_path = self.plot_dag_evolution()
            if fig4_path:
                generated_files.append(fig4_path)
            
            # Figure 5: Duality gap comparison (for methods that have it)
            fig5_path = self.plot_duality_gap()
            if fig5_path:
                generated_files.append(fig5_path)
            
            # Figure 6: Statistical comparison bar chart
            fig6_path = self.plot_statistical_comparison()
            if fig6_path:
                generated_files.append(fig6_path)
            
            print(f"Generated {len(generated_files)} figures in {self.output_dir}")
            
        except Exception as e:
            print(f"Error generating figures: {e}")
        
        return generated_files
    
    def plot_learning_curves(self) -> Optional[Path]:
        """Plot learning curves for all methods"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Learning Curves Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['success_rate', 'avg_cost', 'cost_guarantee_rate', 'avg_reward']
        metric_labels = ['Success Rate', 'Average Cost', 'Cost Guarantee Rate', 'Average Reward']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            
            for method, data in self.results.items():
                if 'seeds' not in data:
                    continue
                
                # Aggregate validation history across seeds
                all_histories = []
                for seed_data in data['seeds'].values():
                    if 'val_history' in seed_data:
                        history = seed_data['val_history']
                        values = [h.get(metric, 0) for h in history]
                        episodes = [h.get('episode', i * 100) for i, h in enumerate(history)]
                        all_histories.append((episodes, values))
                
                if not all_histories:
                    continue
                
                # Compute mean and confidence intervals
                max_len = max(len(h[1]) for h in all_histories)
                aligned_values = []
                
                for episodes, values in all_histories:
                    # Interpolate to common length
                    if len(values) < max_len:
                        # Pad with last value
                        values = values + [values[-1]] * (max_len - len(values))
                    aligned_values.append(values[:max_len])
                
                if aligned_values:
                    mean_values = np.mean(aligned_values, axis=0)
                    std_values = np.std(aligned_values, axis=0)
                    episodes = all_histories[0][0][:max_len]
                    
                    color = self.method_colors.get(method, 'gray')
                    
                    # Plot mean line
                    ax.plot(episodes, mean_values, 
                           label=method.replace('_', '-').upper(), 
                           color=color, linewidth=2)
                    
                    # Plot confidence interval
                    ax.fill_between(episodes, 
                                   mean_values - std_values, 
                                   mean_values + std_values,
                                   alpha=0.2, color=color)
            
            ax.set_xlabel('Episodes')
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'learning_curves.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_pareto_frontier(self) -> Optional[Path]:
        """Plot Pareto frontier (Success vs Cost trade-off)"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract final test results for each method
        plot_data = []
        
        for method, data in self.results.items():
            if 'aggregated' not in data or 'error' in data['aggregated']:
                continue
                
            agg = data['aggregated']
            success_mean = agg.get('success_rate_mean', 0)
            success_std = agg.get('success_rate_std', 0)
            cost_mean = agg.get('avg_cost_mean', 0)
            cost_std = agg.get('avg_cost_std', 0)
            
            plot_data.append({
                'method': method,
                'success_rate': success_mean,
                'success_std': success_std,
                'avg_cost': cost_mean,
                'cost_std': cost_std
            })
        
        if not plot_data:
            print("No data available for Pareto frontier plot")
            return None
        
        # Plot points with error bars
        for data_point in plot_data:
            method = data_point['method']
            color = self.method_colors.get(method, 'gray')
            
            ax.errorbar(data_point['avg_cost'], data_point['success_rate'],
                       xerr=data_point['cost_std'], yerr=data_point['success_std'],
                       marker='o', markersize=10, label=method.replace('_', '-').upper(),
                       color=color, capsize=5, capthick=2, linewidth=2)
        
        # Identify Pareto frontier
        df = pd.DataFrame(plot_data)
        pareto_points = self._find_pareto_frontier(df[['avg_cost', 'success_rate']].values)
        pareto_df = df.iloc[pareto_points].sort_values('avg_cost')
        
        # Draw Pareto frontier line
        if len(pareto_df) > 1:
            ax.plot(pareto_df['avg_cost'], pareto_df['success_rate'], 
                   'k--', alpha=0.5, linewidth=1, label='Pareto Frontier')
        
        ax.set_xlabel('Average Cost ($)', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Success Rate vs Cost Trade-off', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add budget line if available
        budget = 1.0  # Default budget
        if self.results:
            first_method = list(self.results.keys())[0]
            budget = self.results[first_method].get('config', {}).get('budget', 1.0)
        
        ax.axvline(x=budget * 1.05, color='red', linestyle=':', alpha=0.7, 
                   label=f'Budget Limit ({budget * 1.05:.2f})')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'pareto_frontier.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _find_pareto_frontier(self, points: np.ndarray) -> List[int]:
        """Find Pareto frontier points (minimize cost, maximize success)"""
        # Convert to minimization problem (negate success rate)
        points = points.copy()
        points[:, 1] = -points[:, 1]  # Negate success rate
        
        pareto_points = []
        n_points = len(points)
        
        for i in range(n_points):
            is_pareto = True
            for j in range(n_points):
                if i != j:
                    # Check if point j dominates point i
                    if (points[j, 0] <= points[i, 0] and  # Cost
                        points[j, 1] <= points[i, 1] and  # Negative success rate
                        (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1])):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_points.append(i)
        
        return pareto_points
    
    def plot_cost_guarantees(self) -> Optional[Path]:
        """Plot cost guarantee satisfaction over time"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Cost guarantee rate over time
        ax1 = axes[0]
        
        for method, data in self.results.items():
            if 'seeds' not in data:
                continue
            
            # Aggregate cost guarantee rates
            all_cgr_histories = []
            for seed_data in data['seeds'].values():
                if 'val_history' in seed_data:
                    history = seed_data['val_history']
                    cgr_values = [h.get('cost_guarantee_rate', 0) for h in history]
                    episodes = [h.get('episode', i * 100) for i, h in enumerate(history)]
                    all_cgr_histories.append((episodes, cgr_values))
            
            if all_cgr_histories:
                # Align and average
                max_len = max(len(h[1]) for h in all_cgr_histories)
                aligned_values = []
                
                for episodes, values in all_cgr_histories:
                    if len(values) < max_len:
                        values = values + [values[-1]] * (max_len - len(values))
                    aligned_values.append(values[:max_len])
                
                mean_values = np.mean(aligned_values, axis=0)
                episodes = all_cgr_histories[0][0][:max_len]
                
                color = self.method_colors.get(method, 'gray')
                ax1.plot(episodes, mean_values, 
                        label=method.replace('_', '-').upper(), 
                        color=color, linewidth=2)
        
        ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, 
                   label='Target (95%)')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Cost Guarantee Rate')
        ax1.set_title('Cost Guarantee Satisfaction Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Right plot: Final cost guarantee rates with confidence intervals
        ax2 = axes[1]
        
        methods = []
        cgr_means = []
        cgr_stds = []
        colors = []
        
        for method, data in self.results.items():
            if 'aggregated' not in data or 'error' in data['aggregated']:
                continue
            
            agg = data['aggregated']
            cgr_mean = agg.get('cost_guarantee_rate_mean', 0)
            cgr_std = agg.get('cost_guarantee_rate_std', 0)
            
            methods.append(method.replace('_', '-').upper())
            cgr_means.append(cgr_mean)
            cgr_stds.append(cgr_std)
            colors.append(self.method_colors.get(method, 'gray'))
        
        if methods:
            bars = ax2.bar(methods, cgr_means, yerr=cgr_stds, 
                          color=colors, alpha=0.7, capsize=5)
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, cgr_means, cgr_stds):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                        f'{mean:.2%}', ha='center', va='bottom')
            
            ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, 
                       label='Target (95%)')
            ax2.set_ylabel('Cost Guarantee Rate')
            ax2.set_title('Final Cost Guarantee Performance')
            ax2.legend()
            ax2.set_ylim([0, 1.1])
            
            # Rotate x-axis labels if needed
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'cost_guarantees.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_dag_evolution(self) -> Optional[Path]:
        """Plot DAG complexity evolution over time"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DAG Complexity Evolution', fontsize=16, fontweight='bold')
        
        dag_metrics = ['nodes', 'edges', 'diameter', 'clustering']
        metric_labels = ['Average Nodes', 'Average Edges', 'Average Diameter', 'Average Clustering']
        
        for idx, (metric, label) in enumerate(zip(dag_metrics, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            
            for method, data in self.results.items():
                if 'seeds' not in data:
                    continue
                
                # Extract DAG metrics from training history
                all_dag_histories = []
                for seed_data in data['seeds'].values():
                    if 'train_history' in seed_data:
                        history = seed_data['train_history']
                        # Extract DAG metrics if available
                        dag_values = []
                        episodes = []
                        
                        for i, h in enumerate(history):
                            if 'dag_metrics' in h and h['dag_metrics']:
                                dag_values.append(h['dag_metrics'].get(metric, 0))
                                episodes.append(h.get('episode', i))
                        
                        if dag_values:
                            all_dag_histories.append((episodes, dag_values))
                
                if all_dag_histories:
                    # Use a moving average for smoothing
                    window_size = max(1, len(all_dag_histories[0][1]) // 20)
                    
                    for episodes, values in all_dag_histories:
                        if len(values) >= window_size:
                            smoothed = self._moving_average(values, window_size)
                            smoothed_episodes = episodes[window_size-1:]
                            
                            color = self.method_colors.get(method, 'gray')
                            ax.plot(smoothed_episodes, smoothed, 
                                   color=color, alpha=0.3, linewidth=1)
                    
                    # Plot overall average
                    if len(all_dag_histories[0][1]) >= window_size:
                        avg_values = np.mean([self._moving_average(vals, window_size) 
                                            for eps, vals in all_dag_histories], axis=0)
                        avg_episodes = all_dag_histories[0][0][window_size-1:len(avg_values)+window_size-1]
                        
                        ax.plot(avg_episodes, avg_values,
                               label=method.replace('_', '-').upper(),
                               color=self.method_colors.get(method, 'gray'),
                               linewidth=2)
            
            ax.set_xlabel('Episodes')
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'dag_evolution.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _moving_average(self, values: List[float], window_size: int) -> np.ndarray:
        """Compute moving average"""
        return np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    
    def plot_duality_gap(self) -> Optional[Path]:
        """Plot duality gap for PPO-Lagrangian method"""
        
        # Check if any method has duality gap data
        has_duality_gap = False
        for method, data in self.results.items():
            if method == 'ppo_lagrangian' and 'aggregated' in data:
                if 'avg_duality_gap' in data['aggregated']:
                    has_duality_gap = True
                    break
        
        if not has_duality_gap:
            print("No duality gap data available")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Duality gap over time
        ax1 = axes[0]
        
        method = 'ppo_lagrangian'
        if method in self.results and 'seeds' in self.results[method]:
            all_gap_histories = []
            for seed_data in self.results[method]['seeds'].values():
                if 'train_history' in seed_data:
                    # Extract duality gap from method_info if available
                    history = seed_data['train_history']
                    gap_values = []
                    episodes = []
                    
                    for h in history:
                        if 'method_info' in h and 'duality_gap' in h['method_info']:
                            gap_values.append(h['method_info']['duality_gap'])
                            episodes.append(h.get('episode', len(episodes)))
                    
                    if gap_values:
                        all_gap_histories.append((episodes, gap_values))
            
            if all_gap_histories:
                # Plot individual seeds
                for episodes, values in all_gap_histories:
                    ax1.plot(episodes, values, alpha=0.3, color='blue', linewidth=1)
                
                # Plot average
                max_len = min(len(h[1]) for h in all_gap_histories)
                if max_len > 0:
                    avg_values = np.mean([vals[:max_len] for eps, vals in all_gap_histories], axis=0)
                    avg_episodes = all_gap_histories[0][0][:max_len]
                    
                    ax1.plot(avg_episodes, avg_values, 
                            label='PPO-Lagrangian', color='blue', linewidth=2)
        
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Duality Gap')
        ax1.set_title('Duality Gap Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for duality gap
        
        # Right: Final duality gap comparison
        ax2 = axes[1]
        
        # Compare final duality gaps across methods that have them
        methods_with_gaps = []
        final_gaps = []
        
        for method, data in self.results.items():
            if 'aggregated' in data and 'avg_duality_gap' in data['aggregated']:
                methods_with_gaps.append(method.replace('_', '-').upper())
                final_gaps.append(data['aggregated']['avg_duality_gap'])
        
        if methods_with_gaps:
            bars = ax2.bar(methods_with_gaps, final_gaps, 
                          color=[self.method_colors.get(m.lower().replace('-', '_'), 'gray') 
                                for m in methods_with_gaps],
                          alpha=0.7)
            
            ax2.set_ylabel('Final Duality Gap')
            ax2.set_title('Final Duality Gap Comparison')
            ax2.set_yscale('log')
            
            # Add value labels
            for bar, gap in zip(bars, final_gaps):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{gap:.2e}', ha='center', va='bottom', rotation=90)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'duality_gap.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_statistical_comparison(self) -> Optional[Path]:
        """Plot statistical comparison of methods"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['success_rate_mean', 'avg_cost_mean', 'cost_guarantee_rate_mean', 'avg_reward_mean']
        metric_labels = ['Success Rate', 'Average Cost', 'Cost Guarantee Rate', 'Average Reward']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            
            # Extract data for this metric
            methods = []
            values = []
            errors = []
            colors = []
            
            for method, data in self.results.items():
                if 'aggregated' not in data or 'error' in data['aggregated']:
                    continue
                
                agg = data['aggregated']
                if metric in agg:
                    methods.append(method.replace('_', '-').upper())
                    values.append(agg[metric])
                    
                    # Get standard error
                    std_metric = metric.replace('_mean', '_std')
                    std_val = agg.get(std_metric, 0)
                    n_seeds = agg.get('num_seeds', 1)
                    se = std_val / np.sqrt(n_seeds) if n_seeds > 0 else 0
                    errors.append(se)
                    
                    colors.append(self.method_colors.get(method, 'gray'))
            
            if methods:
                bars = ax.bar(methods, values, yerr=errors, 
                             color=colors, alpha=0.7, capsize=5)
                
                # Add significance indicators (simplified)
                self._add_significance_indicators(ax, methods, values, errors)
                
                # Add value labels
                for bar, val, err in zip(bars, values, errors):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., 
                           height + err + (max(values) - min(values)) * 0.02,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=10)
                
                ax.set_ylabel(label)
                ax.set_title(f'{label} Comparison')
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'statistical_comparison.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _add_significance_indicators(self, ax, methods: List[str], 
                                   values: List[float], errors: List[float]):
        """Add significance indicators to bar chart"""
        
        if len(methods) < 2:
            return
        
        # Find best performing method
        if 'Cost' in ax.get_title():
            best_idx = np.argmin(values)  # Lower is better for cost
        else:
            best_idx = np.argmax(values)  # Higher is better for others
        
        # Add star to best performer
        best_bar = ax.patches[best_idx]
        height = best_bar.get_height() + errors[best_idx]
        ax.text(best_bar.get_x() + best_bar.get_width()/2., 
               height + (max(values) - min(values)) * 0.05,
               '★', ha='center', va='bottom', fontsize=16, color='gold')


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate research figures")
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--output', type=str, default='figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        return
    
    # Generate figures
    generator = FigureGenerator(args.results, args.output)
    generated_files = generator.generate_all_figures()
    
    print(f"\nGenerated {len(generated_files)} figures:")
    for filepath in generated_files:
        print(f"  - {filepath}")


if __name__ == "__main__":
    main()