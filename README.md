# WebArena MAS Wrapper

A cost-aware Multi-Agent System (MAS) wrapper for WebArena benchmark that adds comprehensive monitoring for costs, communication, and performance metrics while preserving WebArena's original evaluation methodology.

## Overview

This wrapper integrates WebArena benchmark with our Multi-Agent System (MAS) for cost-aware web task automation. It provides:

- **Cost Monitoring**: Track LLM calls, vision API usage, and action execution costs
- **Communication Monitoring**: Monitor inter-agent communications and patterns
- **Performance Metrics**: Detailed trajectory analysis and success rate tracking
- **Agent Orchestration**: RL-based agent selection for optimal task performance

## Architecture

```
WebArena Environment
        ↓
┌─────────────────┐
│ WebArena Wrapper│ ← Cost Monitor
│                 │ ← Communication Monitor  
│                 │ ← Action Monitor
└────────┬────────┘
         ↓
┌─────────────────┐
│   MAS System    │
│  ┌───────────┐  │
│  │RL Policy  │  │ → Selects agents based on state
│  └───────────┘  │
│  ┌───────────┐  │
│  │  Agents   │  │ → Navigation, Form, Verification
│  └───────────┘  │
└─────────────────┘
```

## Core Components

### 1. WebArenaWrapper
Main wrapper class that orchestrates evaluation with comprehensive monitoring.

### 2. Agent Types
- **NavigationAgent**: Specialized for finding and clicking elements
- **FormFillerAgent**: Handles form completion and data entry  
- **VerificationAgent**: Checks if goals are achieved
- **VisionAnalysisAgent**: Analyzes screenshots for interactive elements

### 3. Monitoring Systems
- **WebArenaCostMonitor**: Tracks different types of costs (LLM, vision, actions)
- **CommunicationMonitor**: Monitors inter-agent communications
- **ActionMonitor**: Logs agent actions and their costs

### 4. Orchestration
- **RLOrchestrator**: Reinforcement learning-based agent selection

## Installation

```bash
# Clone WebArena
git clone https://github.com/web-arena-x/webarena.git
cd webarena
pip install -e .

# Install additional dependencies for MAS
pip install openai anthropic tiktoken networkx

# Clone this wrapper
git clone https://github.com/JoshuaCongHu/webArena-Wrapper.git
cd webArena-Wrapper
```

## Quick Start

```python
from webarena_wrapper import WebArenaWrapper
from agents import NavigationAgent, FormFillerAgent, VerificationAgent
from WebArenaMAS import WebArenaMAS
from orchestrator import RLOrchestrator

# Initialize wrapper
wrapper = WebArenaWrapper()

# Create specialized agents
agents = [
    NavigationAgent(model='gpt-4'),
    FormFillerAgent(model='gpt-3.5'),
    VerificationAgent(model='claude-3')
]

# Create MAS with RL orchestrator
mas = WebArenaMAS(agents=agents, orchestrator=RLOrchestrator())

# Load and run tasks
task_configs = load_webarena_tasks('test_set.json')
results = wrapper.evaluate_mas(mas, task_configs[:10])

# Analyze results
print(f"Success rate: {results['success_rate']:.2%}")
print(f"Average cost: ${results['avg_cost']:.3f}")
print(f"Average steps: {results['avg_steps']:.1f}")
```

## Features

### Cost Tracking
```python
# Track different types of costs
cost_monitor.track_llm_call(model='gpt-4', tokens_in=150, tokens_out=50)
cost_monitor.track_vision_analysis(screenshot_size=1024*1024)

# Get cost breakdown
breakdown = cost_monitor.get_breakdown()
# {'llm_calls': 0.045, 'vision_api_calls': 0.01, 'total': 0.055}
```

### Agent Configuration
```python
# Configure agents with specific strategies
navigation_agent = NavigationAgent(
    model='gpt-4',
    strategies=['xpath', 'css_selector', 'text_match']
)

form_agent = FormFillerAgent(
    model='gpt-3.5',
    validation=True  # Validates form data before submission
)

verification_agent = VerificationAgent(
    model='claude-3',
    screenshot_analysis=True
)
```

## Metrics Tracked

### Performance Metrics
- Success rate
- Steps to completion  
- Time per task
- Backtrack rate

### Cost Metrics
- Total cost per task
- Cost per agent
- Cost per action type
- Token usage breakdown

### Communication Metrics
- Messages per task
- Communication patterns
- Agent collaboration frequency
- Message necessity scores

## Best Practices

1. **Start Small**: Test on 5-10 tasks before full evaluation
2. **Monitor Budget**: Set hard budget limits for expensive models
3. **Cache Results**: WebArena tasks can be slow, cache when possible
4. **Analyze Failures**: Failed actions are learning opportunities
5. **Compare Baselines**: Always compare against single-agent baselines

## Contributing

To add new features:
1. Create feature branch
2. Add tests in `tests/`
3. Update documentation
4. Submit PR with results

## Citation

If using this wrapper, please cite:

```bibtex
@software{mas_webarena_wrapper,
  title={Cost-Aware Multi-Agent System for WebArena},
  author={Joshua Cong Hu},
  year={2024},
  url={https://github.com/JoshuaCongHu/webArena-Wrapper}
}
```

## License

MIT License - see LICENSE file for details.