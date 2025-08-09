#!/usr/bin/env python3
"""
Ablation study framework for WebArena MAS research
"""

import json
import numpy as np
import torch
import argparse
import time
from typing import Dict, List, Any, Optional
import os
import sys
from pathlib import Path
import logging
from itertools import product

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mas.enhanced_webarena_mas import EnhancedWebArenaMAS
from experiments.run_comparison import create_synthetic_tasks, train_mas, evaluate, setup_logging

# Try to import wandb for experiment tracking
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class AblationStudy:
    """Framework for conducting ablation studies"""
    
    def __init__(self, base_method: str = 'p3o', output_dir: str = 'ablation_results'):
        self.base_method = base_method
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Define ablation configurations
        self.ablation_configs = self._define_ablations()
        
    def _define_ablations(self) -> Dict[str, Dict[str, Any]]:
        """Define all ablation study configurations"""
        
        ablations = {
            # Core component ablations
            'full_system': {
                'description': 'Full system with all components',
                'config': {}
            },
            
            'no_orchestrator': {
                'description': 'Disable hierarchical orchestrator',
                'config': {'use_orchestrator': False}
            },
            
            'no_parallel_execution': {
                'description': 'Force sequential execution only',
                'config': {'parallel_execution': False}
            },
            
            'reduced_agents': {
                'description': 'Use only 2 agents instead of 4',
                'config': {'num_agents': 2}
            },
            
            'single_agent': {
                'description': 'Single agent baseline',
                'config': {'num_agents': 1, 'use_orchestrator': False}
            },
            
            # Algorithm-specific ablations for P3O
            'no_adaptive_penalty': {
                'description': 'Disable adaptive penalty coefficient',
                'config': {'algorithm_config': {'adaptive_penalty': False}}
            },
            
            'high_penalty': {
                'description': 'Use high penalty coefficient (100x)',
                'config': {'algorithm_config': {'penalty_coef': 1000.0}}
            },
            
            'low_penalty': {
                'description': 'Use low penalty coefficient (0.1x)',
                'config': {'algorithm_config': {'penalty_coef': 1.0}}
            },
            
            # Budget variations
            'tight_budget': {
                'description': 'Tighter budget constraint (0.5x)',
                'config': {'budget': 0.5}
            },
            
            'loose_budget': {
                'description': 'Looser budget constraint (2x)',
                'config': {'budget': 2.0}
            },
            
            # Network architecture ablations
            'small_networks': {
                'description': 'Smaller neural networks (64 dim)',
                'config': {'state_dim': 64, 'action_dim': 32}
            },
            
            'large_networks': {
                'description': 'Larger neural networks (256 dim)',
                'config': {'state_dim': 256, 'action_dim': 128}
            },
            
            # Learning rate ablations
            'high_lr': {
                'description': 'High learning rate (10x)',
                'config': {'algorithm_config': {'lr_policy': 3e-3, 'lr_critic': 3e-3}}
            },
            
            'low_lr': {
                'description': 'Low learning rate (0.1x)',
                'config': {'algorithm_config': {'lr_policy': 3e-5, 'lr_critic': 3e-5}}
            }
        }
        
        # Add method-specific ablations
        if self.base_method == 'ppo_lagrangian':
            ablations.update({
                'no_lambda_adaptation': {
                    'description': 'Fixed Lagrange multiplier',
                    'config': {'algorithm_config': {'lr_lambda': 0.0}}
                },
                
                'high_lambda_lr': {
                    'description': 'High lambda learning rate',
                    'config': {'algorithm_config': {'lr_lambda': 1e-2}}
                }
            })
        
        elif self.base_method == 'macpo':
            ablations.update({
                'no_coordination': {
                    'description': 'Disable coordination penalty',
                    'config': {'algorithm_config': {'coordination_weight': 0.0}}
                },
                
                'high_coordination': {
                    'description': 'High coordination penalty',
                    'config': {'algorithm_config': {'coordination_weight': 1.0}}
                }
            })
        
        return ablations
    
    def run_all_ablations(self, 
                         tasks: List[Dict],
                         base_config: Dict,
                         seeds: List[int] = [42, 1337, 2024]) -> Dict[str, Any]:
        """Run all ablation studies"""
        
        self.logger.info(f"Starting ablation study for {self.base_method}")
        self.logger.info(f"Running {len(self.ablation_configs)} ablations with {len(seeds)} seeds each")
        
        results = {
            'base_method': self.base_method,
            'base_config': base_config,
            'ablations': {},
            'summary': {}
        }
        
        # Initialize wandb if available
        if HAS_WANDB:
            wandb.init(
                project=f"mas-webarena-ablations-{self.base_method}",
                config={
                    'base_method': self.base_method,
                    'num_ablations': len(self.ablation_configs),
                    'seeds': seeds
                }
            )
        
        for ablation_name, ablation_spec in self.ablation_configs.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Running ablation: {ablation_name}")
            self.logger.info(f"Description: {ablation_spec['description']}")
            self.logger.info('='*60)
            
            try:
                ablation_results = self._run_single_ablation(
                    ablation_name, ablation_spec, tasks, base_config, seeds
                )
                results['ablations'][ablation_name] = ablation_results
                
                # Save intermediate results
                self._save_intermediate_results(ablation_name, ablation_results)
                
            except Exception as e:
                self.logger.error(f"Ablation {ablation_name} failed: {e}")
                results['ablations'][ablation_name] = {'error': str(e)}
        
        # Generate summary
        results['summary'] = self._generate_ablation_summary(results['ablations'])
        
        # Save complete results
        results_file = self.output_dir / f'{self.base_method}_ablation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Ablation study completed. Results saved to {results_file}")
        
        if HAS_WANDB:
            wandb.finish()
        
        return results
    
    def _run_single_ablation(self,
                           ablation_name: str,
                           ablation_spec: Dict,
                           tasks: List[Dict],
                           base_config: Dict,
                           seeds: List[int]) -> Dict[str, Any]:
        """Run a single ablation across multiple seeds"""
        
        ablation_results = {
            'name': ablation_name,
            'description': ablation_spec['description'],
            'config': ablation_spec['config'],
            'seeds': {},
            'aggregated': {}
        }
        
        for seed_idx, seed in enumerate(seeds):
            self.logger.info(f"Running seed {seed} for {ablation_name} ({seed_idx + 1}/{len(seeds)})")
            
            # Set random seeds
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            # Merge configs
            merged_config = self._merge_configs(base_config, ablation_spec['config'])
            
            # Create MAS with ablated configuration
            mas = self._create_ablated_mas(merged_config)
            
            # Train and evaluate
            seed_results = train_mas(mas, tasks, merged_config, seed)
            ablation_results['seeds'][seed] = seed_results
            
            # Log to wandb if available
            if HAS_WANDB:
                if 'test_results' in seed_results:
                    test_results = seed_results['test_results']
                    wandb.log({
                        f'{ablation_name}/success_rate': test_results.get('success_rate', 0),
                        f'{ablation_name}/cost_guarantee_rate': test_results.get('cost_guarantee_rate', 0),
                        f'{ablation_name}/avg_cost': test_results.get('avg_cost', 0),
                        'seed': seed
                    })
        
        # Aggregate results across seeds
        ablation_results['aggregated'] = self._aggregate_ablation_results(ablation_results['seeds'])
        
        return ablation_results
    
    def _merge_configs(self, base_config: Dict, ablation_config: Dict) -> Dict:
        """Merge base config with ablation config"""
        merged = base_config.copy()
        
        for key, value in ablation_config.items():
            if key == 'algorithm_config' and key in merged:
                # Merge algorithm configs
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        
        return merged
    
    def _create_ablated_mas(self, config: Dict) -> EnhancedWebArenaMAS:
        """Create MAS instance with ablated configuration"""
        
        # Extract algorithm config
        algorithm_config = config.pop('algorithm_config', {})
        
        # Create MAS
        mas = EnhancedWebArenaMAS(method=self.base_method, **config)
        
        # Apply algorithm-specific configurations
        if algorithm_config and hasattr(mas.algorithm, '__dict__'):
            for key, value in algorithm_config.items():
                if hasattr(mas.algorithm, key):
                    setattr(mas.algorithm, key, value)
                    self.logger.debug(f"Set {key} = {value} on algorithm")
        
        return mas
    
    def _aggregate_ablation_results(self, seed_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate results across seeds for a single ablation"""
        
        if not seed_results:
            return {'error': 'No seed results'}
        
        # Extract test results
        test_metrics = []
        for seed_data in seed_results.values():
            if 'test_results' in seed_data and seed_data['test_results']:
                test_metrics.append(seed_data['test_results'])
        
        if not test_metrics:
            return {'error': 'No test results'}
        
        aggregated = {}
        
        # Aggregate metrics
        for key in test_metrics[0].keys():
            if key not in ['error'] and isinstance(test_metrics[0][key], (int, float)):
                values = [m[key] for m in test_metrics if key in m]
                if values:
                    aggregated[f'{key}_mean'] = np.mean(values)
                    aggregated[f'{key}_std'] = np.std(values)
                    aggregated[f'{key}_ci'] = 1.96 * np.std(values) / np.sqrt(len(values))
        
        aggregated['num_seeds'] = len(test_metrics)
        
        return aggregated
    
    def _save_intermediate_results(self, ablation_name: str, results: Dict):
        """Save intermediate results for this ablation"""
        filepath = self.output_dir / f'{ablation_name}_results.json'
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _generate_ablation_summary(self, ablation_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate summary of ablation study results"""
        
        summary = {
            'num_ablations': len(ablation_results),
            'successful_ablations': 0,
            'failed_ablations': 0,
            'performance_comparison': {},
            'key_findings': []
        }
        
        # Collect performance data
        performance_data = {}
        baseline_performance = None
        
        for ablation_name, results in ablation_results.items():
            if 'aggregated' in results and 'error' not in results['aggregated']:
                summary['successful_ablations'] += 1
                
                agg = results['aggregated']
                performance_data[ablation_name] = {
                    'success_rate': agg.get('success_rate_mean', 0),
                    'cost_guarantee_rate': agg.get('cost_guarantee_rate_mean', 0),
                    'avg_cost': agg.get('avg_cost_mean', 0)
                }
                
                # Use full_system as baseline
                if ablation_name == 'full_system':
                    baseline_performance = performance_data[ablation_name]
            else:
                summary['failed_ablations'] += 1
        
        summary['performance_comparison'] = performance_data
        
        # Generate key findings
        if baseline_performance:
            summary['key_findings'] = self._generate_key_findings(
                performance_data, baseline_performance
            )
        
        return summary
    
    def _generate_key_findings(self, 
                             performance_data: Dict[str, Dict], 
                             baseline: Dict[str, float]) -> List[str]:
        """Generate key findings from ablation results"""
        
        findings = []
        
        # Find best and worst performing ablations
        success_rates = {name: data['success_rate'] 
                        for name, data in performance_data.items() 
                        if name != 'full_system'}
        
        if success_rates:
            best_success = max(success_rates.items(), key=lambda x: x[1])
            worst_success = min(success_rates.items(), key=lambda x: x[1])
            
            baseline_success = baseline['success_rate']
            
            # Best performing ablation
            if best_success[1] > baseline_success:
                improvement = ((best_success[1] - baseline_success) / baseline_success) * 100
                findings.append(
                    f"{best_success[0]} improves success rate by {improvement:.1f}% "
                    f"({best_success[1]:.3f} vs {baseline_success:.3f})"
                )
            
            # Most harmful ablation
            if worst_success[1] < baseline_success:
                degradation = ((baseline_success - worst_success[1]) / baseline_success) * 100
                findings.append(
                    f"{worst_success[0]} reduces success rate by {degradation:.1f}% "
                    f"({worst_success[1]:.3f} vs {baseline_success:.3f})"
                )
        
        # Analyze cost guarantee performance
        cgr_data = {name: data['cost_guarantee_rate'] 
                   for name, data in performance_data.items() 
                   if name != 'full_system'}
        
        if cgr_data:
            baseline_cgr = baseline['cost_guarantee_rate']
            high_cgr_ablations = [name for name, cgr in cgr_data.items() 
                                 if cgr > baseline_cgr + 0.05]  # 5% improvement
            
            if high_cgr_ablations:
                findings.append(
                    f"Ablations improving cost guarantee rate: {', '.join(high_cgr_ablations)}"
                )
        
        # Component importance analysis
        critical_components = []
        for name, data in performance_data.items():
            if ('no_' in name and 
                data['success_rate'] < baseline['success_rate'] * 0.9):  # 10% drop
                component = name.replace('no_', '').replace('_', ' ')
                critical_components.append(component)
        
        if critical_components:
            findings.append(
                f"Critical components (>10% performance drop when removed): "
                f"{', '.join(critical_components)}"
            )
        
        return findings


def generate_ablation_report(results: Dict[str, Any], output_file: str = None):
    """Generate a comprehensive ablation study report"""
    
    if output_file is None:
        output_file = f"ablation_report_{results['base_method']}.md"
    
    report_lines = [
        f"# Ablation Study Report: {results['base_method'].upper()}",
        "",
        f"**Base Method:** {results['base_method']}",
        f"**Total Ablations:** {results['summary']['num_ablations']}",
        f"**Successful:** {results['summary']['successful_ablations']}",
        f"**Failed:** {results['summary']['failed_ablations']}",
        "",
        "## Key Findings",
        ""
    ]
    
    # Add key findings
    for finding in results['summary']['key_findings']:
        report_lines.append(f"- {finding}")
    
    report_lines.extend(["", "## Performance Comparison", ""])
    
    # Create performance table
    if results['summary']['performance_comparison']:
        header = "| Ablation | Success Rate | Cost Guarantee Rate | Avg Cost |"
        separator = "|----------|-------------|-------------------|----------|"
        report_lines.extend([header, separator])
        
        # Sort by success rate (descending)
        sorted_ablations = sorted(
            results['summary']['performance_comparison'].items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )
        
        for name, data in sorted_ablations:
            row = (f"| {name.replace('_', ' ').title()} | "
                   f"{data['success_rate']:.3f} | "
                   f"{data['cost_guarantee_rate']:.3f} | "
                   f"{data['avg_cost']:.3f} |")
            report_lines.append(row)
    
    report_lines.extend(["", "## Detailed Results", ""])
    
    # Add detailed results for each ablation
    for name, ablation_data in results['ablations'].items():
        if 'error' in ablation_data:
            report_lines.extend([
                f"### {name.replace('_', ' ').title()}",
                f"**Status:** Failed - {ablation_data['error']}",
                ""
            ])
        else:
            agg = ablation_data['aggregated']
            report_lines.extend([
                f"### {name.replace('_', ' ').title()}",
                f"**Description:** {ablation_data['description']}",
                f"**Success Rate:** {agg.get('success_rate_mean', 0):.3f} ± {agg.get('success_rate_std', 0):.3f}",
                f"**Cost Guarantee Rate:** {agg.get('cost_guarantee_rate_mean', 0):.3f} ± {agg.get('cost_guarantee_rate_std', 0):.3f}",
                f"**Average Cost:** {agg.get('avg_cost_mean', 0):.3f} ± {agg.get('avg_cost_std', 0):.3f}",
                f"**Seeds:** {agg.get('num_seeds', 0)}",
                ""
            ])
    
    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Ablation report written to {output_file}")


def main():
    """Main function for running ablation studies"""
    
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument('--base_method', type=str, default='p3o',
                       choices=['p3o', 'ppo_lagrangian', 'macpo'],
                       help='Base method to ablate')
    parser.add_argument('--ablations', type=str, default='all',
                       help='Comma-separated list of ablations to run (or "all")')
    parser.add_argument('--seeds', type=str, default='42,1337',
                       help='Comma-separated list of random seeds')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of training episodes per seed')
    parser.add_argument('--num_tasks', type=int, default=500,
                       help='Number of synthetic tasks to generate')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                       help='Output directory for results')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with fewer episodes')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    if args.quick_test:
        episodes = 50
        num_tasks = 100
        logger.info("Running in quick test mode")
    else:
        episodes = args.episodes
        num_tasks = args.num_tasks
    
    # Base configuration
    base_config = {
        'episodes': episodes,
        'budget': 1.0,
        'num_agents': 4,
        'device': 'cpu',
        'eval_interval': max(10, episodes // 10),
        'save_interval': max(50, episodes // 5)
    }
    
    logger.info(f"Running ablation study for {args.base_method}")
    logger.info(f"Base config: {base_config}")
    logger.info(f"Seeds: {seeds}")
    
    # Generate tasks
    logger.info(f"Generating {num_tasks} synthetic tasks...")
    tasks = create_synthetic_tasks(num_tasks)
    
    # Run ablation study
    ablation_study = AblationStudy(args.base_method, args.output_dir)
    
    # Filter ablations if specified
    if args.ablations != 'all':
        ablation_names = [name.strip() for name in args.ablations.split(',')]
        ablation_study.ablation_configs = {
            name: config for name, config in ablation_study.ablation_configs.items()
            if name in ablation_names
        }
        logger.info(f"Running specific ablations: {ablation_names}")
    
    # Run study
    results = ablation_study.run_all_ablations(tasks, base_config, seeds)
    
    # Generate report
    report_file = f"{args.output_dir}/ablation_report_{args.base_method}.md"
    generate_ablation_report(results, report_file)
    
    logger.info("Ablation study completed!")


if __name__ == "__main__":
    main()