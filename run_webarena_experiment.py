# run_webarena_experiment.py
def main():
    # Initialize WebArena wrapper
    wrapper = WebArenaWrapper()
    
    # Create specialized agents for web tasks

    #later modify agents needed
    agents = [
        NavigationAgent(model='gpt-4'),
        FormFillerAgent(model='gpt-3.5'),
        VerificationAgent(model='claude-3'),
        VisionAnalysisAgent(model='gpt-4-vision')  # For screenshots
    ]
    
    # Create MAS with RL orchestrator
    mas = WebArenaMAS(
        agents=agents,
        orchestrator=RLOrchestrator()
    )
    
    # Load WebArena tasks
    task_configs = load_webarena_tasks('test_set.json')
    
    # Run evaluation
    results = wrapper.evaluate_mas(mas, task_configs[:10])  # Start small
    
    # Analyze results
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Average cost per task: ${results['avg_cost']:.3f}")
    print(f"Average steps per task: {results['avg_steps']:.1f}")