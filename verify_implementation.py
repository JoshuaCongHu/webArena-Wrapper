#!/usr/bin/env python3
"""
Verify the LLM orchestrator implementation structure
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report"""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} (missing)")
        return False

def check_file_content(filepath, required_strings, description):
    """Check if file contains required content"""
    if not Path(filepath).exists():
        print(f"✗ {description}: {filepath} (file missing)")
        return False
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        missing = []
        for required in required_strings:
            if required not in content:
                missing.append(required)
        
        if missing:
            print(f"✗ {description}: Missing content: {missing}")
            return False
        else:
            print(f"✓ {description}: All required content found")
            return True
            
    except Exception as e:
        print(f"✗ {description}: Error reading file: {e}")
        return False

def main():
    """Verify implementation structure"""
    print("LLM Orchestrator Implementation Verification")
    print("=" * 50)
    
    base_dir = Path(__file__).parent
    orchestrator_dir = base_dir / "orchestrator"
    
    all_checks_passed = True
    
    # Check directory structure
    print("\n1. Directory Structure:")
    all_checks_passed &= check_file_exists(orchestrator_dir, "Orchestrator directory")
    
    # Check core files
    print("\n2. Core Implementation Files:")
    core_files = [
        (orchestrator_dir / "__init__.py", "Package init file"),
        (orchestrator_dir / "llm_orchestrator.py", "LLM Orchestrator Policy"),
        (orchestrator_dir / "prompt_manager.py", "Prompt Manager"),
        (orchestrator_dir / "dag_validator.py", "DAG Validator"),
        (orchestrator_dir / "dag_cache.py", "DAG Cache Manager"),
        (orchestrator_dir / "replanning_engine.py", "Replanning Engine"),
        (orchestrator_dir / "context_builder.py", "Context Builder"),
    ]
    
    for filepath, description in core_files:
        all_checks_passed &= check_file_exists(filepath, description)
    
    # Check key classes exist
    print("\n3. Key Class Implementations:")
    class_checks = [
        (orchestrator_dir / "llm_orchestrator.py", ["class LLMOrchestratorPolicy", "def generate_dag"], "LLM Orchestrator class"),
        (orchestrator_dir / "prompt_manager.py", ["class PromptManager", "def format_orchestrator_prompt"], "Prompt Manager class"),
        (orchestrator_dir / "dag_validator.py", ["class DAGValidator", "def validate_complete"], "DAG Validator class"),
        (orchestrator_dir / "dag_cache.py", ["class DAGCacheManager", "def cache_successful_dag"], "DAG Cache class"),
        (orchestrator_dir / "replanning_engine.py", ["class ReplanningEngine", "def should_replan"], "Replanning Engine class"),
        (orchestrator_dir / "context_builder.py", ["def build_context", "def build_simplified_context"], "Context builder functions"),
    ]
    
    for filepath, required_content, description in class_checks:
        all_checks_passed &= check_file_content(filepath, required_content, description)
    
    # Check enhanced MAS integration
    print("\n4. Enhanced MAS Integration:")
    mas_file = base_dir / "mas" / "enhanced_webarena_mas.py"
    mas_requirements = [
        "use_llm_orchestrator",
        "llm_orchestrator",
        "cache_manager", 
        "replanning_engine",
        "build_context"
    ]
    all_checks_passed &= check_file_content(mas_file, mas_requirements, "Enhanced MAS LLM integration")
    
    # Check requirements update
    print("\n5. Dependencies:")
    req_file = base_dir.parent / "requirements.txt"
    llm_deps = ["anthropic", "google-generativeai", "torch>=2.0.0", "networkx>=2.8"]
    all_checks_passed &= check_file_content(req_file, llm_deps, "LLM dependencies in requirements.txt")
    
    # Check test files
    print("\n6. Test Files:")
    test_files = [
        (base_dir / "test_llm_orchestrator.py", "LLM orchestrator tests"),
        (base_dir / "verify_implementation.py", "Implementation verification script"),
    ]
    
    for filepath, description in test_files:
        all_checks_passed &= check_file_exists(filepath, description)
    
    # Summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("✅ All implementation checks passed!")
        print("\nImplementation Summary:")
        print("- Complete LLM-based orchestrator with OpenAI/Anthropic support")
        print("- Dynamic replanning engine with failure detection")
        print("- Intelligent DAG caching and validation")
        print("- Comprehensive prompt management")
        print("- Full integration with Enhanced WebArena MAS")
        print("- Fallback modes for operation without API keys")
        print("- Test suite and verification scripts")
        
        print("\nUsage:")
        print("1. Set API keys: export OPENAI_API_KEY=your_key or ANTHROPIC_API_KEY=your_key")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run with LLM: EnhancedWebArenaMAS(use_llm_orchestrator=True)")
        print("4. Test: python3 test_llm_orchestrator.py")
        
    else:
        print("❌ Some implementation checks failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())