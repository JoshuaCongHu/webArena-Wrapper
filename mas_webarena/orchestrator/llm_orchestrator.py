import json
import time
import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import hashlib

@dataclass
class LLMConfig:
    """Configuration for LLM"""
    model_name: str = "gpt-4-turbo"  # or "claude-3-opus", "gemini-1.5-pro"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    api_key: str = None
    api_type: str = "openai"  # or "anthropic", "google"

class LLMOrchestratorPolicy:
    """LLM-based orchestrator for DAG generation"""
    
    def __init__(
        self,
        llm_model: str = "gpt-4-turbo",
        method: str = "p3o",
        budget: float = 1.0,
        max_nodes: int = 10,
        num_agents: int = 4,
        state_dim: int = 1024,
        action_dim: int = 64,
        device: str = "cpu"
    ):
        self.llm_config = LLMConfig(model_name=llm_model)
        self.method = method
        self.budget = budget
        self.max_nodes = max_nodes
        self.num_agents = num_agents
        self.device = torch.device(device)
        
        # Initialize LLM client
        self.llm_client = self._initialize_llm_client()
        
        # Initialize prompt manager
        from .prompt_manager import PromptManager
        self.prompt_manager = PromptManager()
        
        # Initialize validator
        from .dag_validator import DAGValidator
        self.validator = DAGValidator(max_nodes, num_agents)
        
        # For RL training - encode LLM outputs to work with existing algorithms
        self.state_encoder = torch.nn.Linear(state_dim, 256)
        self.action_decoder = torch.nn.Linear(256, action_dim)
        
        # Tracking
        self.generation_history = []
        self.validation_failures = []
        
    def _initialize_llm_client(self):
        """Initialize the appropriate LLM client"""
        if self.llm_config.api_type == "openai":
            try:
                import openai
                api_key = self.llm_config.api_key or os.getenv("OPENAI_API_KEY")
                if api_key:
                    openai.api_key = api_key
                    return openai
                else:
                    print("Warning: No OpenAI API key found. LLM orchestrator will use fallback mode.")
                    return None
            except ImportError:
                print("Warning: OpenAI library not installed. LLM orchestrator will use fallback mode.")
                return None
        elif self.llm_config.api_type == "anthropic":
            try:
                import anthropic
                api_key = self.llm_config.api_key or os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    return anthropic.Anthropic(api_key=api_key)
                else:
                    print("Warning: No Anthropic API key found. LLM orchestrator will use fallback mode.")
                    return None
            except ImportError:
                print("Warning: Anthropic library not installed. LLM orchestrator will use fallback mode.")
                return None
        return None
        
    def generate_dag(
        self, 
        context: Dict[str, Any],
        use_cache: bool = True,
        cache_manager: Optional['DAGCacheManager'] = None
    ) -> Tuple[Dict, torch.Tensor]:
        """
        Generate DAG using LLM with logits for RL training
        
        Args:
            context: Task context including state, constraints, history
            use_cache: Whether to check cache first
            cache_manager: Optional cache manager instance
            
        Returns:
            dag_json: Generated DAG in JSON format
            logits: Tensor of logits for RL training
        """
        # Check cache first
        if use_cache and cache_manager:
            cached_dag = cache_manager.get_cached_dag(context)
            if cached_dag:
                # Return cached DAG with dummy logits
                return cached_dag, torch.zeros(1, 256)
        
        # Build prompt
        prompt = self.prompt_manager.format_orchestrator_prompt(
            context=context,
            method=self.method,
            budget=self.budget,
            constraint_info=self._get_constraint_info(context)
        )
        
        # Generate with LLM
        try:
            if self.llm_client and self.llm_config.api_type == "openai":
                response = self.llm_client.ChatCompletion.create(
                    model=self.llm_config.model_name,
                    messages=[
                        {"role": "system", "content": "You are a task orchestrator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens,
                    logprobs=True  # Request logprobs for RL
                )
                
                text_response = response.choices[0].message.content
                # Extract logprobs (simplified - actual implementation would be more complex)
                logprobs = response.choices[0].logprobs
                logits = self._logprobs_to_logits(logprobs)
                
            elif self.llm_client and self.llm_config.api_type == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.llm_config.model_name,
                    max_tokens=self.llm_config.max_tokens,
                    temperature=self.llm_config.temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                text_response = response.content[0].text
                logits = torch.zeros(1, 256)  # Anthropic doesn't provide logprobs
                
            else:
                # Fallback to rule-based generation
                text_response = self._generate_fallback_response(context)
                logits = torch.zeros(1, 256)
                
            # Parse JSON response
            dag_json = self.prompt_manager.parse_llm_response(text_response)
            
            # Validate DAG
            is_valid, errors = self.validator.validate_complete(dag_json, context)
            if not is_valid:
                self.validation_failures.append({
                    'dag': dag_json,
                    'errors': errors,
                    'context': context
                })
                # Generate fallback DAG
                dag_json = self.validator.generate_fallback_dag(context)
                
        except Exception as e:
            print(f"LLM generation failed: {e}")
            dag_json = self.validator.generate_fallback_dag(context)
            logits = torch.zeros(1, 256)
            
        # Store in history
        self.generation_history.append({
            'context': context,
            'dag': dag_json,
            'timestamp': time.time()
        })
        
        return dag_json, logits
    
    def _logprobs_to_logits(self, logprobs) -> torch.Tensor:
        """Convert log probabilities to logits tensor for RL"""
        # Simplified conversion - actual implementation would handle token sequences
        if logprobs and hasattr(logprobs, 'content') and len(logprobs.content) > 0:
            # Extract numerical values and convert to tensor
            values = []
            for token_logprob in logprobs.content[:256]:
                if hasattr(token_logprob, 'logprob'):
                    values.append(token_logprob.logprob)
                else:
                    values.append(0.0)
            
            # Pad or truncate to 256
            while len(values) < 256:
                values.append(0.0)
            values = values[:256]
            
            return torch.tensor(values).unsqueeze(0)
        return torch.zeros(1, 256)
    
    def _get_constraint_info(self, context: Dict) -> Dict:
        """Extract constraint-specific information based on method"""
        info = {
            'budget': self.budget,
            'cost_spent': context.get('constraints', {}).get('cost_spent', 0),
            'remaining_budget': self.budget - context.get('constraints', {}).get('cost_spent', 0)
        }
        
        if self.method == "ppo_lagrangian":
            info['duality_gap'] = context.get('learning_info', {}).get('duality_gap', 0)
            info['lambda'] = context.get('learning_info', {}).get('lambda', 0)
        elif self.method == "p3o":
            info['penalty_coef'] = context.get('learning_info', {}).get('penalty_coef', 10.0)
        elif self.method == "macpo":
            info['lambda_coord'] = context.get('learning_info', {}).get('lambda_coord', 0.5)
            
        return info
    
    def _generate_fallback_response(self, context: Dict) -> str:
        """Generate a fallback response when LLM is not available"""
        # Simple rule-based DAG generation
        task_type = context.get('task', {}).get('intent', 'unknown')
        num_steps = min(context.get('task', {}).get('expected_steps', 3), self.max_nodes)
        
        # Create sequential DAG
        nodes = []
        edges = []
        agent_assignments = {}
        
        for i in range(num_steps):
            if i == 0:
                task_name = "navigate_to_page"
                cost = 0.02
            elif i == 1:
                task_name = "find_search_element"
                cost = 0.03
            elif i == 2:
                task_name = "enter_search_query"
                cost = 0.04
            else:
                task_name = f"action_step_{i}"
                cost = 0.05
                
            nodes.append({
                "id": i,
                "task": task_name,
                "description": f"Step {i+1} for {task_type}",
                "estimated_cost": cost,
                "difficulty": 0.3 + i * 0.1,
                "priority": "high" if i < 2 else "medium",
                "expected_reward": 0.2
            })
            
            if i > 0:
                edges.append([i-1, i])
                
            # Assign cheapest agent for simple tasks
            agent_assignments[str(i)] = "kimi-k2" if cost < 0.04 else "gpt-4-turbo"
        
        # Create parallel groups (sequential for fallback)
        parallel_groups = [[i] for i in range(num_steps)]
        
        fallback_dag = {
            "dag": {
                "nodes": nodes,
                "edges": edges,
                "agent_assignments": agent_assignments,
                "parallel_groups": parallel_groups,
                "checkpoint_nodes": [num_steps // 2] if num_steps > 2 else [],
                "confidence": 0.6,
                "estimated_total_cost": sum(node["estimated_cost"] for node in nodes),
                "reasoning": "Fallback rule-based generation used due to LLM unavailability"
            }
        }
        
        return json.dumps(fallback_dag, indent=2)