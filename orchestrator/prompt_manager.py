import json
from typing import Dict, List, Any

class PromptManager:
    """Manage LLM prompts and parsing"""
    
    def __init__(self):
        self.prompt_templates = self._load_prompt_templates()
        
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates"""
        return {
            'orchestrator': """You are a web automation orchestrator managing multiple agents with cost constraints.

TASK CONTEXT:
{task_json}

CONSTRAINTS:
- Budget: ${budget:.2f} (Remaining: ${remaining_budget:.2f})
- Method: {method}
- Must satisfy: P(cost ≤ {alpha} × budget) ≥ {beta}
{method_specific}

AVAILABLE AGENTS:
{agents_json}

CURRENT STATE:
- Page: {page_type}
- Available elements: {elements}
- Previous actions: {previous_actions}

REQUIREMENTS:
1. Decompose the task into specific subtasks (nodes)
2. Define execution order with edges
3. Assign the most cost-effective agent for each subtask
4. Identify parallel execution opportunities
5. Set checkpoints after critical steps
6. Total estimated cost must be < remaining budget

OUTPUT FORMAT:
{{
    "dag": {{
        "nodes": [
            {{
                "id": 0,
                "task": "navigate_to_search",
                "description": "Navigate to search page",
                "estimated_cost": 0.02,
                "difficulty": 0.3,
                "priority": "high",
                "expected_reward": 0.2
            }}
        ],
        "edges": [[0, 1], [1, 2]],
        "agent_assignments": {{"0": "kimi-k2", "1": "gpt-4"}},
        "parallel_groups": [[0], [1, 2]],
        "checkpoint_nodes": [1],
        "confidence": 0.85,
        "estimated_total_cost": 0.15,
        "reasoning": "Using cheap agents for simple tasks, parallel execution for independent subtasks"
    }}
}}

Generate the JSON DAG:""",

            'replanning': """The current plan has failed and needs modification.

FAILURE CONTEXT:
- Reason: {reason}
- Failed at node: {failure_point}
- Completed nodes: {completed_nodes}
- Remaining budget: ${remaining_budget:.2f}

ORIGINAL PLAN:
{original_dag}

CURRENT STATE:
{current_state}

Generate a new plan for the remaining subtasks that avoids the previous failure.
Focus on:
1. Learning from the failure reason
2. Using more reliable agents if budget allows
3. Adding verification steps
4. Keeping completed work intact

Use the same JSON format as before:"""
        }
    
    def format_orchestrator_prompt(
        self,
        context: Dict,
        method: str,
        budget: float,
        constraint_info: Dict
    ) -> str:
        """Format the orchestrator prompt with context"""
        
        # Extract key information
        task_json = json.dumps(context.get('task', {}), indent=2)
        agents_json = self._format_agents(context.get('agent_pool', {}))
        
        # Method-specific information
        method_specific = self._format_method_specific(method, constraint_info)
        
        prompt = self.prompt_templates['orchestrator'].format(
            task_json=task_json,
            budget=budget,
            remaining_budget=constraint_info.get('remaining_budget', budget),
            method=method,
            alpha=context.get('constraints', {}).get('alpha', 1.05),
            beta=context.get('constraints', {}).get('beta', 0.95),
            method_specific=method_specific,
            agents_json=agents_json,
            page_type=context.get('current_state', {}).get('page_type', 'unknown'),
            elements=', '.join(context.get('current_state', {}).get('elements_available', [])),
            previous_actions=', '.join(context.get('current_state', {}).get('previous_actions', []))
        )
        
        return prompt
    
    def format_replanning_prompt(
        self,
        context: Dict,
        reason: str,
        failure_point: int,
        completed_nodes: List[int],
        original_dag: Dict,
        remaining_budget: float
    ) -> str:
        """Format the replanning prompt with failure context"""
        
        current_state_json = json.dumps(context.get('current_state', {}), indent=2)
        original_dag_json = json.dumps(original_dag, indent=2)
        
        prompt = self.prompt_templates['replanning'].format(
            reason=reason,
            failure_point=failure_point,
            completed_nodes=', '.join(map(str, completed_nodes)),
            remaining_budget=remaining_budget,
            original_dag=original_dag_json,
            current_state=current_state_json
        )
        
        return prompt
    
    def _format_agents(self, agent_pool: Dict) -> str:
        """Format agent information for prompt"""
        lines = []
        for name, info in agent_pool.items():
            if info.get('availability', False):
                lines.append(f"- {name}: cost=${info['cost_per_action']:.3f}, "
                           f"tags={info['tags']}, success_rate={info.get('success_rate', 0.8):.2f}")
        return '\n'.join(lines) if lines else "No agents available"
    
    def _format_method_specific(self, method: str, info: Dict) -> str:
        """Format method-specific constraint information"""
        if method == "ppo_lagrangian":
            return f"- Duality Gap: {info.get('duality_gap', 0):.4f}\n- Lambda: {info.get('lambda', 0):.4f}"
        elif method == "p3o":
            return f"- Penalty Coefficient: {info.get('penalty_coef', 10.0):.2f}"
        elif method == "macpo":
            return f"- Coordination Lambda: {info.get('lambda_coord', 0.5):.4f}"
        return ""
    
    def parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response to extract DAG JSON"""
        try:
            # Try to extract JSON from response
            # Handle different response formats
            if '```json' in response:
                # Extract JSON from markdown code block
                start = response.find('```json') + 7
                end = response.find('```', start)
                json_str = response[start:end].strip()
            elif '```' in response and '{' in response:
                # Extract JSON from any code block
                start = response.find('```')
                start = response.find('{', start)
                end = response.rfind('```')
                if end == -1:
                    end = len(response)
                json_str = response[start:end].strip()
                # Remove trailing ```
                if json_str.endswith('```'):
                    json_str = json_str[:-3].strip()
            elif '{' in response:
                # Find JSON object
                start = response.find('{')
                # Find matching closing brace
                depth = 0
                end = start
                for i, char in enumerate(response[start:], start):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                json_str = response[start:end]
            else:
                raise ValueError("No JSON found in response")
                
            # Parse JSON
            dag_json = json.loads(json_str)
            
            # Ensure required structure
            if 'dag' not in dag_json:
                dag_json = {'dag': dag_json}
                
            return dag_json
            
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response: {response[:500]}...")
            
            # Return minimal fallback structure
            return {
                'dag': {
                    'nodes': [
                        {
                            'id': 0,
                            'task': 'fallback_action',
                            'description': 'Fallback action due to parsing failure',
                            'estimated_cost': 0.05,
                            'difficulty': 0.5,
                            'priority': 'medium',
                            'expected_reward': 0.1
                        }
                    ],
                    'edges': [],
                    'agent_assignments': {'0': 'gpt-4-turbo'},
                    'parallel_groups': [[0]],
                    'checkpoint_nodes': [],
                    'confidence': 0.3,
                    'estimated_total_cost': 0.05,
                    'reasoning': 'Fallback due to LLM response parsing failure'
                }
            }