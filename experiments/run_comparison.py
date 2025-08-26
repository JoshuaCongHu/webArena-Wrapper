#!/usr/bin/env python3
"""
Main experiment script for comparing constraint methods in WebArena MAS
"""

import json
import numpy as np
import torch
import argparse
import time
from typing import Dict, List, Tuple
import os
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mas.enhanced_webarena_mas import EnhancedWebArenaMAS

# Try to import wandb for experiment tracking
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not available, experiment tracking disabled")


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiment.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_synthetic_tasks(num_tasks: int = 1000, seed: int = 42) -> List[Dict]:
    """Create synthetic WebArena tasks for experiments"""
    np.random.seed(seed)
    
    tasks = []
    
    # Task templates
    templates = [
        {
            'intent': 'Navigate to {site} and search for {item}',
            'sites': ['shopping.com', 'amazon.com', 'ebay.com'],
            'difficulty': 'easy'
        },
        {
            'intent': 'Login to {site} and update profile information',
            'sites': ['reddit.com', 'gitlab.com', 'maps.google.com'],
            'difficulty': 'medium'
        },
        {
            'intent': 'Book a {service} on {site} for {date}',
            'sites': ['classifieds.com', 'shopping.com'],
            'difficulty': 'hard'
        },
        {
            'intent': 'Compare prices for {item} across multiple sites',
            'sites': ['shopping.com', 'amazon.com'],
            'difficulty': 'medium'
        },
        {
            'intent': 'Complete checkout process for items in cart',
            'sites': ['shopping.com'],
            'difficulty': 'hard'
        }
    ]
    
    items = ['laptop', 'book', 'phone', 'headphones', 'tablet', 'watch']
    services = ['appointment', 'reservation', 'meeting', 'consultation']
    dates = ['tomorrow', 'next week', 'Friday', 'weekend']
    
    for i in range(num_tasks):
        template = templates[i % len(templates)]
        
        # Fill in template variables
        intent = template['intent'].format(
            site=np.random.choice(template['sites']),
            item=np.random.choice(items),
            service=np.random.choice(services),
            date=np.random.choice(dates)
        )
        
        task = {
            'task_id': f'task_{i:04d}',
            'intent': intent,
            'sites': template['sites'][:np.random.randint(1, len(template['sites']) + 1)],
            'difficulty': template['difficulty'],
            'expected_steps': np.random.randint(3, 8),
            'template_id': templates.index(template)
        }
        
        tasks.append(task)
    
    return tasks


def run_experiment(method: str, 
                  tasks: List[Dict],
                  config: Dict,
                  seeds: List[int] = [42, 1337, 2024]) -> Dict:
    """Run experiment for one method across multiple seeds"""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment for method: {method}")
    
    results = {
        'method': method,
        'config': config,
        'seeds': {},
        'aggregated': {}
    }
    
    for seed_idx, seed in enumerate(seeds):
        logger.info(f"Running seed {seed} ({seed_idx + 1}/{len(seeds)})")
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Initialize MAS
        mas_config = {
            'method': method,
            'budget': config.get('budget', 1.0),
            'num_agents': config.get('num_agents', 4),
            'state_dim': config.get('state_dim', 128),
            'action_dim': config.get('action_dim', 64),
            'device': config.get('device', 'cpu')
        }
        
        mas = EnhancedWebArenaMAS(**mas_config)
        
        # Training loop
        seed_results = train_mas(mas, tasks, config, seed)
        results['seeds'][seed] = seed_results
        
        logger.info(f"Completed seed {seed}")
    
    # Aggregate across seeds
    results['aggregated'] = aggregate_results(results['seeds'])
    logger.info(f"Experiment completed for method: {method}")
    
    return results


def train_mas(mas: EnhancedWebArenaMAS, 
              tasks: List[Dict],
              config: Dict,
              seed: int) -> Dict:
    """Training loop with evaluation"""
    
    logger = logging.getLogger(__name__)
    episodes = config.get('episodes', 1000)
    eval_interval = config.get('eval_interval', 100)
    save_interval = config.get('save_interval', 500)
    
    # Split tasks
    train_split = config.get('train_split', 0.6)
    val_split = config.get('val_split', 0.2)
    
    n_train = int(len(tasks) * train_split)
    n_val = int(len(tasks) * val_split)
    
    train_tasks = tasks[:n_train]
    val_tasks = tasks[n_train:n_train + n_val]
    test_tasks = tasks[n_train + n_val:]
    
    results = {
        'train_history': [],
        'val_history': [],
        'test_results': None,
        'seed': seed,
        'config': config
    }
    
    start_time = time.time()
    
    for episode in range(episodes):
        # Sample training task
        task = train_tasks[np.random.randint(len(train_tasks))]
        
        # Execute task
        try:
            outcome = mas.solve_task(task)
            
            # Log to wandb if available
            if HAS_WANDB:
                wandb.log({
                    f'{mas.method}/success': outcome['success'],
                    f'{mas.method}/cost': outcome['cost'],
                    f'{mas.method}/reward': outcome['reward'],
                    f'{mas.method}/episode': episode,
                    **{f'{mas.method}/{k}': v for k, v in outcome['method_info'].items() 
                       if isinstance(v, (int, float))}
                }, step=episode)
            
            # Store training outcome
            results['train_history'].append({
                'episode': episode,
                'success': outcome['success'],
                'cost': outcome['cost'],
                'reward': outcome['reward'],
                'dag_nodes': outcome['dag_metrics'].get('nodes', 0),
                'method_info': outcome['method_info']  # For duality gap visualization
            })
            
        except Exception as e:
            logger.warning(f"Episode {episode} failed: {e}")
            continue
        
        # Validation evaluation
        if episode % eval_interval == 0 and episode > 0:
            val_metrics = evaluate(mas, val_tasks[:20], f"Validation episode {episode}")
            results['val_history'].append({
                'episode': episode,
                **val_metrics
            })
            
            logger.info(
                f"Episode {episode}: "
                f"Success={val_metrics['success_rate']:.2%}, "
                f"Cost=${val_metrics['avg_cost']:.3f}, "
                f"CGR={val_metrics['cost_guarantee_rate']:.2%}"
            )
        
        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            checkpoint_path = f"checkpoints/{mas.method}_seed{seed}_ep{episode}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            mas.save_checkpoint(checkpoint_path)
    
    # Final test evaluation
    logger.info("Running final test evaluation...")
    results['test_results'] = evaluate(mas, test_tasks, "Final test")
    results['training_time'] = time.time() - start_time
    
    return results


def evaluate(mas: EnhancedWebArenaMAS, 
            tasks: List[Dict],
            description: str = "") -> Dict:
    """Evaluate MAS on task set"""
    
    logger = logging.getLogger(__name__)
    if description:
        logger.info(f"Starting evaluation: {description}")
    
    outcomes = []
    failed_tasks = 0
    
    for i, task in enumerate(tasks):
        try:
            outcome = mas.solve_task(task)
            outcomes.append(outcome)
            
            if i % 10 == 0:
                logger.debug(f"Evaluated {i + 1}/{len(tasks)} tasks")
                
        except Exception as e:
            logger.warning(f"Task {task.get('task_id', i)} failed: {e}")
            failed_tasks += 1
    
    if not outcomes:
        logger.error("No successful task evaluations!")
        return {'error': 'No successful evaluations'}
    
    # Compute metrics
    metrics = {
        'success_rate': np.mean([o['success'] for o in outcomes]),
        'avg_cost': np.mean([o['cost'] for o in outcomes]),
        'avg_reward': np.mean([o['reward'] for o in outcomes]),
        'cost_guarantee_rate': np.mean([o['cost'] <= mas.budget * 1.05 for o in outcomes]),
        'avg_dag_complexity': np.mean([o['dag_metrics'].get('nodes', 0) for o in outcomes]),
        'failed_tasks': failed_tasks,
        'total_tasks': len(tasks)
    }
    
    # Add method-specific metrics
    if mas.method == 'ppo_lagrangian':
        duality_gaps = [o['method_info'].get('duality_gap', 0) for o in outcomes]
        metrics['avg_duality_gap'] = np.mean(duality_gaps)
    
    if description:
        logger.info(f"Evaluation complete: {description}")
    
    return metrics


def aggregate_results(seed_results: Dict[str, Dict]) -> Dict:
    """Aggregate results across seeds"""
    
    if not seed_results:
        return {}
    
    # Get all metric keys from test results
    test_metrics = []
    for seed_data in seed_results.values():
        if 'test_results' in seed_data and seed_data['test_results']:
            test_metrics.append(seed_data['test_results'])
    
    if not test_metrics:
        return {'error': 'No test results to aggregate'}
    
    aggregated = {}
    
    # Aggregate test metrics
    for key in test_metrics[0].keys():
        if key not in ['error'] and isinstance(test_metrics[0][key], (int, float)):
            values = [m[key] for m in test_metrics if key in m]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_ci'] = 1.96 * np.std(values) / np.sqrt(len(values))  # 95% CI
    
    # Add sample size
    aggregated['num_seeds'] = len(test_metrics)
    aggregated['total_training_time'] = sum(
        seed_data.get('training_time', 0) for seed_data in seed_results.values()
    )
    
    return aggregated


def generate_results_table(all_results: Dict[str, Dict]):
    """Generate results table for paper"""
    
    logger = logging.getLogger(__name__)
    logger.info("Generating results table...")
    
    table_rows = []
    header = "| Method | Success↑ | Cost↓ | CGR↑ | Time(s)↓ |"
    separator = "|--------|----------|-------|------|----------|"
    
    table_rows.append(header)
    table_rows.append(separator)
    
    for method, results in all_results.items():
        if 'aggregated' not in results or 'error' in results['aggregated']:
            continue
            
        agg = results['aggregated']
        
        row = f"| {method} | {agg.get('success_rate_mean', 0):.3f}±{agg.get('success_rate_std', 0):.3f} | " \
              f"{agg.get('avg_cost_mean', 0):.3f}±{agg.get('avg_cost_std', 0):.3f} | " \
              f"{agg.get('cost_guarantee_rate_mean', 0):.3f}±{agg.get('cost_guarantee_rate_std', 0):.3f} | " \
              f"{agg.get('total_training_time', 0)/agg.get('num_seeds', 1):.1f} |"
        
        table_rows.append(row)
    
    table = "\n".join(table_rows)
    
    # Save table
    with open('results/results_table.md', 'w') as f:
        f.write("# Experimental Results\n\n")
        f.write(table)
        f.write("\n\n")
        f.write("- Success↑: Success rate (higher is better)\n")
        f.write("- Cost↓: Average cost per episode (lower is better)\n") 
        f.write("- CGR↑: Cost guarantee rate (higher is better)\n")
        f.write("- Time(s)↓: Average training time per seed in seconds (lower is better)\n")
    
    logger.info("Results table saved to results/results_table.md")
    print(f"\n{table}\n")


def main():
    """Main experiment runner"""
    
    parser = argparse.ArgumentParser(description="Run WebArena MAS experiments")
    parser.add_argument('--methods', type=str, default='p3o,ppo_lagrangian,macpo',
                       help='Comma-separated list of methods to compare')
    parser.add_argument('--seeds', type=str, default='42,1337,2024',
                       help='Comma-separated list of random seeds')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes per seed')
    parser.add_argument('--num_tasks', type=int, default=1000,
                       help='Number of synthetic tasks to generate')
    parser.add_argument('--budget', type=float, default=1.0,
                       help='Budget constraint for experiments')
    parser.add_argument('--num_agents', type=int, default=4,
                       help='Number of agents in the system')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run experiments on (cpu/cuda)')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with fewer episodes')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.log_level)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Parse arguments
    methods = [m.strip() for m in args.methods.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    if args.quick_test:
        episodes = 50
        num_tasks = 100
        logger.info("Running in quick test mode")
    else:
        episodes = args.episodes
        num_tasks = args.num_tasks
    
    # Configuration
    config = {
        'episodes': episodes,
        'budget': args.budget,
        'num_agents': args.num_agents,
        'device': args.device,
        'eval_interval': max(10, episodes // 10),
        'save_interval': max(50, episodes // 5)
    }
    
    logger.info(f"Experiment configuration: {config}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Seeds: {seeds}")
    
    # Initialize wandb if available
    if HAS_WANDB:
        wandb.init(
            project="mas-webarena-neurips",
            config={
                'methods': methods,
                'seeds': seeds,
                **config
            }
        )
    
    # Generate tasks
    logger.info(f"Generating {num_tasks} synthetic tasks...")
    tasks = create_synthetic_tasks(num_tasks)
    
    # Save tasks
    with open(f'{args.output_dir}/tasks.json', 'w') as f:
        json.dump(tasks, f, indent=2)
    
    # Run experiments
    all_results = {}
    
    for method in methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment for method: {method}")
        logger.info('='*60)
        
        try:
            results = run_experiment(method, tasks, config, seeds)
            all_results[method] = results
            
            # Save intermediate results
            method_file = f'{args.output_dir}/{method}_results.json'
            with open(method_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Experiment failed for method {method}: {e}")
            continue
    
    # Generate final outputs
    if all_results:
        logger.info("Generating final results...")
        
        # Save all results
        with open(f'{args.output_dir}/comparison_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Generate results table
        generate_results_table(all_results)
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        for method, results in all_results.items():
            if 'aggregated' in results and 'error' not in results['aggregated']:
                agg = results['aggregated']
                print(f"\n{method}:")
                print(f"  Success Rate: {agg.get('success_rate_mean', 0):.1%} ± {agg.get('success_rate_std', 0):.1%}")
                print(f"  Cost Guarantee Rate: {agg.get('cost_guarantee_rate_mean', 0):.1%} ± {agg.get('cost_guarantee_rate_std', 0):.1%}")
                print(f"  Average Cost: {agg.get('avg_cost_mean', 0):.3f} ± {agg.get('avg_cost_std', 0):.3f}")
        
        logger.info(f"All results saved to {args.output_dir}/")
    else:
        logger.error("No successful experiments completed!")
    
    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()