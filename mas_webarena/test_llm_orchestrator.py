#!/usr/bin/env python3
"""
Test cases for LLM orchestrator components
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator import (
    LLMOrchestratorPolicy,
    PromptManager,
    DAGValidator,
    DAGCacheManager,
    ReplanningEngine,
    build_simplified_context,
    validate_context
)
from utils.budget_tracker import BudgetTracker

def test_prompt_manager():
    """Test prompt manager functionality"""
    print("Testing PromptManager...")
    
    manager = PromptManager()
    
    # Test context building
    context = build_simplified_context(
        task_intent="Search for flights from NYC to LA",
        page_type="homepage",
        elements=["search_box", "date_picker", "submit_button"],
        budget_remaining=0.8,
        method="p3o"
    )
    
    # Test prompt formatting
    prompt = manager.format_orchestrator_prompt(
        context=context,
        method="p3o",
        budget=1.0,
        constraint_info={'remaining_budget': 0.8, 'penalty_coef': 10.0}
    )
    
    assert len(prompt) > 100, "Prompt should be substantial"
    assert "Search for flights" in prompt, "Task intent should be in prompt"
    assert "p3o" in prompt, "Method should be in prompt"
    assert "0.80" in prompt, "Budget should be in prompt"
    
    print("✓ PromptManager tests passed")

def test_dag_validator():
    """Test DAG validator functionality"""
    print("Testing DAGValidator...")
    
    validator = DAGValidator(max_nodes=5, num_agents=3)
    
    # Test valid DAG
    valid_dag = {
        'dag': {
            'nodes': [
                {'id': 0, 'task': 'navigate', 'estimated_cost': 0.02},
                {'id': 1, 'task': 'search', 'estimated_cost': 0.03},
                {'id': 2, 'task': 'select', 'estimated_cost': 0.04}
            ],
            'edges': [[0, 1], [1, 2]],
            'agent_assignments': {'0': 'agent1', '1': 'agent2', '2': 'agent1'}
        }
    }
    
    context = build_simplified_context("test task", budget_remaining=1.0)
    is_valid, errors = validator.validate_complete(valid_dag, context)
    
    assert is_valid, f"Valid DAG should pass validation, errors: {errors}"
    
    # Test invalid DAG (cycle)
    invalid_dag = {
        'dag': {
            'nodes': [
                {'id': 0, 'task': 'navigate', 'estimated_cost': 0.02},
                {'id': 1, 'task': 'search', 'estimated_cost': 0.03}
            ],
            'edges': [[0, 1], [1, 0]],  # Cycle
            'agent_assignments': {'0': 'agent1', '1': 'agent2'}
        }
    }
    
    is_valid, errors = validator.validate_complete(invalid_dag, context)
    assert not is_valid, "Invalid DAG should fail validation"
    assert any('cycle' in error.lower() for error in errors), "Should detect cycle"
    
    # Test fallback generation
    fallback_dag = validator.generate_fallback_dag(context)
    assert 'dag' in fallback_dag, "Fallback should have DAG structure"
    assert len(fallback_dag['dag']['nodes']) > 0, "Fallback should have nodes"
    
    print("✓ DAGValidator tests passed")

def test_dag_cache_manager():
    """Test DAG cache manager functionality"""
    print("Testing DAGCacheManager...")
    
    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = DAGCacheManager(cache_dir=temp_dir, max_cache_size=10)
        
        # Test context and DAG
        context = build_simplified_context("test search task")
        dag = {
            'dag': {
                'nodes': [{'id': 0, 'task': 'search'}],
                'edges': [],
                'agent_assignments': {'0': 'agent1'}
            }
        }
        metrics = {'success': True, 'cost': 0.1, 'reward': 0.8}
        
        # Test caching
        cache_manager.cache_successful_dag(context, dag, metrics)
        
        # Test retrieval
        cached_dag = cache_manager.get_cached_dag(context)
        assert cached_dag is not None, "Should retrieve cached DAG"
        assert cached_dag['dag']['nodes'][0]['task'] == 'search', "Cached DAG should match"
        
        # Test cache stats
        stats = cache_manager.get_cache_stats()
        assert stats['total_entries'] >= 1, "Should have cached entries"
        assert stats['cache_hits'] >= 1, "Should have cache hits"
        
        print("✓ DAGCacheManager tests passed")

def test_llm_orchestrator():
    """Test LLM orchestrator (fallback mode)"""
    print("Testing LLMOrchestratorPolicy...")
    
    # Test without API keys (should use fallback)
    orchestrator = LLMOrchestratorPolicy(
        llm_model="gpt-4-turbo",
        method="p3o",
        budget=1.0,
        max_nodes=5,
        num_agents=3
    )
    
    context = build_simplified_context("book a flight")
    
    # Generate DAG (should work in fallback mode)
    dag_json, logits = orchestrator.generate_dag(context, use_cache=False)
    
    assert 'dag' in dag_json, "Should return DAG structure"
    assert len(dag_json['dag']['nodes']) > 0, "Should have nodes"
    assert 'reasoning' in dag_json['dag'], "Should have reasoning"
    
    print("✓ LLMOrchestratorPolicy tests passed")

def test_replanning_engine():
    """Test replanning engine functionality"""
    print("Testing ReplanningEngine...")
    
    # Create orchestrator for replanning engine
    orchestrator = LLMOrchestratorPolicy(
        llm_model="gpt-4-turbo",
        method="p3o",
        budget=1.0
    )
    
    engine = ReplanningEngine(
        orchestrator=orchestrator,
        replan_threshold=0.3,
        max_replans_per_task=2
    )
    
    # Test replanning decision
    context = build_simplified_context("test task")
    context['constraints']['cost_spent'] = 0.6  # High cost spent
    context['task']['expected_steps'] = 10
    
    trajectory = [{'success': False}, {'success': False}]  # Failed attempts
    current_dag = {'dag': {'confidence': 0.4}}  # Low confidence
    
    should_replan, reason = engine.should_replan(context, trajectory, current_dag)
    
    assert should_replan, "Should trigger replanning with failures and low confidence"
    assert reason is not None, "Should provide reason for replanning"
    
    # Test replanning execution
    original_dag = {
        'dag': {
            'nodes': [
                {'id': 0, 'task': 'step1'},
                {'id': 1, 'task': 'step2'}
            ],
            'edges': [[0, 1]],
            'agent_assignments': {'0': 'agent1', '1': 'agent2'}
        }
    }
    
    new_dag = engine.execute_replanning(
        current_dag=original_dag,
        current_state=context,
        reason=reason,
        completed_nodes=[0]  # First node completed
    )
    
    assert 'dag' in new_dag, "Should return updated DAG"
    
    print("✓ ReplanningEngine tests passed")

def test_context_validation():
    """Test context validation"""
    print("Testing context validation...")
    
    # Test valid context
    valid_context = build_simplified_context("test task")
    errors = validate_context(valid_context)
    assert len(errors) == 0, f"Valid context should have no errors: {errors}"
    
    # Test invalid context
    invalid_context = {}
    errors = validate_context(invalid_context)
    assert len(errors) > 0, "Invalid context should have errors"
    
    print("✓ Context validation tests passed")

def test_integration():
    """Test integration between components"""
    print("Testing component integration...")
    
    # Create all components
    budget_tracker = BudgetTracker(initial_budget=1.0)
    orchestrator = LLMOrchestratorPolicy(method="p3o", budget=1.0)
    cache_manager = DAGCacheManager(cache_dir="test_cache", max_cache_size=5)
    replanning_engine = ReplanningEngine(orchestrator=orchestrator)
    
    # Build context
    context = build_simplified_context("integration test task")
    
    # Generate DAG
    dag_json, logits = orchestrator.generate_dag(context, use_cache=True, cache_manager=cache_manager)
    
    # Validate DAG
    validator = DAGValidator()
    is_valid, errors = validator.validate_complete(dag_json, context)
    
    if not is_valid:
        print(f"DAG validation errors (expected in fallback mode): {errors}")
    
    # Cache successful result
    cache_manager.cache_successful_dag(
        context=context,
        dag=dag_json,
        metrics={'success': True, 'cost': 0.2, 'reward': 0.8}
    )
    
    # Test cache retrieval
    cached = cache_manager.get_cached_dag(context)
    assert cached is not None, "Should retrieve from cache"
    
    # Cleanup
    cache_dir = Path("test_cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    
    print("✓ Integration tests passed")

def main():
    """Run all tests"""
    print("Running LLM Orchestrator Tests...")
    print("=" * 50)
    
    try:
        test_prompt_manager()
        test_dag_validator()
        test_dag_cache_manager()
        test_llm_orchestrator()
        test_replanning_engine()
        test_context_validation()
        test_integration()
        
        print("=" * 50)
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())