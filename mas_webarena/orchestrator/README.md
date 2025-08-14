# LLM-Based Dynamic Orchestrator

A sophisticated LLM-powered orchestration system for WebArena Multi-Agent Systems with dynamic replanning capabilities and constrained reinforcement learning integration.

## 🏗️ Architecture Overview

```
orchestrator/
├── llm_orchestrator.py      # Core LLM orchestrator with multi-provider support
├── prompt_manager.py        # Advanced prompt engineering and parsing
├── dag_validator.py         # Comprehensive DAG validation system
├── dag_cache.py            # Intelligent caching with similarity matching
├── replanning_engine.py    # Dynamic replanning with failure detection
├── context_builder.py      # Rich context management utilities
└── __init__.py             # Package interface
```

## 🚀 Quick Start

### Basic Usage

```python
from orchestrator import LLMOrchestratorPolicy, build_context
from utils.budget_tracker import BudgetTracker

# Initialize orchestrator
orchestrator = LLMOrchestratorPolicy(
    llm_model="gpt-4-turbo",  # or "claude-3-opus", "gemini-1.5-pro"
    method="p3o",             # "ppo_lagrangian", "p3o", "macpo"
    budget=1.0,
    max_nodes=8,
    num_agents=4
)

# Build context
context = build_context(
    task={'intent': 'Book a flight from NYC to LA', 'sites': ['expedia.com']},
    current_observation="Homepage with search form visible",
    trajectory=[],
    budget_tracker=BudgetTracker(1.0),
    method="p3o"
)

# Generate DAG
dag_json, logits = orchestrator.generate_dag(context)
print(f"Generated {len(dag_json['dag']['nodes'])} steps")
```

### With Enhanced MAS

```python
from mas.enhanced_webarena_mas import EnhancedWebArenaMAS

# Create MAS with LLM orchestrator
mas = EnhancedWebArenaMAS(
    method='p3o',
    budget=2.0,
    use_llm_orchestrator=True,
    llm_model='gpt-4-turbo',
    enable_replanning=True
)

# Solve task with automatic replanning
result = mas.solve_task({
    'intent': 'Find and book cheapest flight',
    'sites': ['kayak.com'],
    'expected_steps': 6
})
```

## 🔧 Core Components

### LLMOrchestratorPolicy

The main orchestrator that generates DAGs using large language models.

**Features:**
- Multi-provider support (OpenAI, Anthropic, Google)
- Automatic fallback to rule-based generation
- RL-compatible logits generation
- Method-aware constraint handling

**Supported LLM Providers:**
- **OpenAI**: GPT-4 Turbo, GPT-4, GPT-3.5 Turbo
- **Anthropic**: Claude-3 Opus, Claude-3 Sonnet, Claude-3 Haiku  
- **Google**: Gemini-1.5 Pro, Gemini-1.0 Pro

### ReplanningEngine

Dynamic replanning system that adapts plans during execution.

**Replanning Triggers:**
- Action failures
- Budget inefficiency
- Constraint violations
- Low confidence scores
- Unexpected state changes

**Example:**
```python
from orchestrator import ReplanningEngine

engine = ReplanningEngine(
    orchestrator=orchestrator,
    replan_threshold=0.3,
    max_replans_per_task=3
)

# Check if replanning needed
should_replan, reason = engine.should_replan(
    current_state=context,
    trajectory=trajectory,
    current_dag=dag
)

if should_replan:
    new_dag = engine.execute_replanning(
        current_dag=dag,
        current_state=context,
        reason=reason,
        completed_nodes=[0, 1]
    )
```

### DAGCacheManager

Intelligent caching system for successful DAGs.

**Features:**
- Context-based cache keys
- Similarity-based retrieval
- Age and success rate validation
- LRU eviction policy

```python
from orchestrator import DAGCacheManager

cache = DAGCacheManager(
    cache_dir="dag_cache",
    max_cache_size=1000
)

# Cache successful DAG
cache.cache_successful_dag(
    context=context,
    dag=dag,
    metrics={'success': True, 'cost': 0.3, 'reward': 0.8}
)

# Retrieve similar DAGs
cached_dag = cache.get_cached_dag(context)
similar_dags = cache.get_similar_dags(context, limit=5)
```

### DAGValidator

Comprehensive validation system ensuring DAG correctness.

**Validation Checks:**
- Structure integrity (nodes, edges, assignments)
- DAG properties (acyclicity, connectivity)
- Cost constraints
- Parallel group validity

```python
from orchestrator import DAGValidator

validator = DAGValidator(max_nodes=10, num_agents=4)

is_valid, errors = validator.validate_complete(dag_json, context)
if not is_valid:
    print(f"Validation errors: {errors}")
    fallback_dag = validator.generate_fallback_dag(context)
```

### PromptManager

Advanced prompt engineering system with method-specific templates.

**Features:**
- Dynamic template generation
- Method-specific constraint formatting
- Robust JSON parsing with fallback
- Replanning-aware prompts

```python
from orchestrator import PromptManager

manager = PromptManager()

# Format orchestrator prompt
prompt = manager.format_orchestrator_prompt(
    context=context,
    method="p3o",
    budget=1.0,
    constraint_info={'penalty_coef': 10.0}
)

# Parse LLM response
dag_json = manager.parse_llm_response(llm_response)
```

## 🎯 Constraint Methods

The orchestrator supports three constrained RL methods:

### P3O (Penalized Proximal Policy Optimization)
```python
orchestrator = LLMOrchestratorPolicy(method="p3o")
# Uses quadratic penalty for constraint violations
```

### PPO-Lagrangian
```python
orchestrator = LLMOrchestratorPolicy(method="ppo_lagrangian")
# Uses dual optimization with Lagrange multipliers
```

### MACPO (Multi-Agent Constrained Policy Optimization)
```python
orchestrator = LLMOrchestratorPolicy(method="macpo")
# Distributed learning with global constraint coordination
```

## 🔧 Configuration

### Environment Variables

```bash
# LLM API Keys (optional - has fallback mode)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

### LLM Configuration

```python
from orchestrator import LLMConfig

config = LLMConfig(
    model_name="gpt-4-turbo",
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
    api_type="openai"
)
```

## 📊 Monitoring & Metrics

### Cache Statistics
```python
stats = cache_manager.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Total entries: {stats['total_entries']}")
```

### Replanning Statistics
```python
replan_stats = replanning_engine.get_replan_stats()
print(f"Total replans: {replan_stats['total_replans']}")
print(f"Common reasons: {replan_stats['most_common_reasons']}")
```

## 🧪 Testing

```bash
# Run comprehensive test suite
python3 test_llm_orchestrator.py

# Verify implementation structure
python3 verify_implementation.py
```

## 🛡️ Error Handling

The system includes comprehensive error handling:

- **LLM Failures**: Automatic fallback to rule-based generation
- **API Rate Limits**: Exponential backoff and retry logic
- **Invalid DAGs**: Validation and automatic correction
- **Budget Constraints**: Hard stops and soft warnings
- **Context Errors**: Graceful degradation with minimal context

## 📈 Performance Optimization

- **Caching**: Intelligent DAG reuse for similar contexts
- **Parallel Execution**: DAG-level parallelization
- **Early Termination**: Budget-aware stopping criteria
- **Batch Processing**: Efficient multi-step execution

## 🔮 Future Extensions

- **Multi-modal Orchestration**: Vision + language models
- **Hierarchical Planning**: Multi-level task decomposition  
- **Federated Orchestration**: Cross-domain coordination
- **Adaptive Learning**: Online prompt optimization

---

For more information, see the main [LLM_ORCHESTRATOR_SUMMARY.md](../LLM_ORCHESTRATOR_SUMMARY.md) documentation.