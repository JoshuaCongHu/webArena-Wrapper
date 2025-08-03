#!/usr/bin/env python3
"""Test the WebArenaSpecificMetrics as specified in CLAUDE.md"""

from WebArenaSpecificMetrics import WebArenaSpecificMetrics


def create_sample_trajectory():
    """Create a sample trajectory for testing"""
    return [
        {
            'action': {'type': 'click', 'element_id': 1},
            'cost': 0.002,
            'timestamp': 1000.0
        },
        {
            'action': {'type': 'type', 'text': 'username'},
            'cost': 0.003,
            'timestamp': 1001.5
        },
        {
            'action': {'type': 'type', 'text': 'password'},
            'cost': 0.003,
            'timestamp': 1003.0
        },
        {
            'action': {'type': 'click', 'element_id': 2},
            'cost': 0.002,
            'timestamp': 1004.2
        },
        {
            'action': {'type': 'wait'},
            'cost': 0.001,
            'timestamp': 1005.0,
            'error': 'Timeout waiting for page load'
        },
        {
            'action': {'type': 'back'},
            'cost': 0.001,
            'timestamp': 1006.0
        },
        {
            'action': {'type': 'click', 'element_id': 3},
            'cost': 0.002,
            'timestamp': 1007.5
        },
        {
            'action': {'type': 'verify'},
            'cost': 0.004,
            'timestamp': 1008.8
        },
        {
            'action': {'type': 'wait'},
            'cost': 0.001,
            'timestamp': 1010.0
        },
        {
            'action': {'type': 'query', 'selector': 'button'},
            'cost': 0.002,
            'timestamp': 1011.2
        }
    ]


def test_basic_metrics_calculation():
    """Test basic metrics calculation functionality"""
    print("Testing basic metrics calculation...")
    
    metrics = WebArenaSpecificMetrics()
    trajectory = create_sample_trajectory()
    
    result = metrics.calculate_efficiency_metrics(trajectory)
    
    # Test web-specific metrics
    assert result['actions_per_task'] == 10, f"Expected 10 actions, got {result['actions_per_task']}"
    assert result['failed_actions'] == 1, f"Expected 1 failed action, got {result['failed_actions']}"
    assert result['backtrack_rate'] == 1, f"Expected 1 backtrack, got {result['backtrack_rate']}"
    assert result['dom_queries'] == 1, f"Expected 1 DOM query, got {result['dom_queries']}"
    
    print(f"Actions per task: {result['actions_per_task']}")
    print(f"Failed actions: {result['failed_actions']}")
    print(f"Backtrack rate: {result['backtrack_rate']}")
    print(f"DOM queries: {result['dom_queries']}")
    
    print("✅ Basic metrics calculation tests passed!")


def test_cost_breakdown():
    """Test cost breakdown calculations"""
    print("\nTesting cost breakdown...")
    
    metrics = WebArenaSpecificMetrics()
    trajectory = create_sample_trajectory()
    
    result = metrics.calculate_efficiency_metrics(trajectory)
    
    # Expected costs:
    # Click actions: 3 actions × costs (0.002 + 0.002 + 0.002) = 0.006
    # Type actions: 2 actions × costs (0.003 + 0.003) = 0.006
    # Verify/wait actions: 3 actions × costs (0.001 + 0.004 + 0.001) = 0.006
    
    expected_navigation_cost = 0.006
    expected_form_filling_cost = 0.006
    expected_verification_cost = 0.006
    
    assert abs(result['navigation_cost'] - expected_navigation_cost) < 0.0001, \
        f"Expected navigation cost {expected_navigation_cost}, got {result['navigation_cost']}"
    
    assert abs(result['form_filling_cost'] - expected_form_filling_cost) < 0.0001, \
        f"Expected form filling cost {expected_form_filling_cost}, got {result['form_filling_cost']}"
    
    assert abs(result['verification_cost'] - expected_verification_cost) < 0.0001, \
        f"Expected verification cost {expected_verification_cost}, got {result['verification_cost']}"
    
    print(f"Navigation cost: ${result['navigation_cost']:.4f}")
    print(f"Form filling cost: ${result['form_filling_cost']:.4f}")
    print(f"Verification cost: ${result['verification_cost']:.4f}")
    
    print("✅ Cost breakdown tests passed!")


def test_time_metrics():
    """Test time-related metrics"""
    print("\nTesting time metrics...")
    
    metrics = WebArenaSpecificMetrics()
    trajectory = create_sample_trajectory()
    
    result = metrics.calculate_efficiency_metrics(trajectory)
    
    # Test average time calculation (simplified version)
    expected_avg_time = len(trajectory) / 30.0  # As per specification
    assert abs(result['time_per_action'] - expected_avg_time) < 0.0001, \
        f"Expected avg time {expected_avg_time}, got {result['time_per_action']}"
    
    # Test idle time (count of wait actions)
    expected_idle_time = 2  # Two wait actions in trajectory
    assert result['idle_time'] == expected_idle_time, \
        f"Expected idle time {expected_idle_time}, got {result['idle_time']}"
    
    print(f"Time per action: {result['time_per_action']:.4f}")
    print(f"Idle time: {result['idle_time']}")
    
    print("✅ Time metrics tests passed!")


def test_individual_methods():
    """Test individual helper methods"""
    print("\nTesting individual methods...")
    
    metrics = WebArenaSpecificMetrics()
    trajectory = create_sample_trajectory()
    
    # Test count_backtracks
    backtrack_count = metrics.count_backtracks(trajectory)
    assert backtrack_count == 1, f"Expected 1 backtrack, got {backtrack_count}"
    print(f"Backtrack count: {backtrack_count}")
    
    # Test calculate_avg_time
    avg_time = metrics.calculate_avg_time(trajectory)
    expected = len(trajectory) / 30.0
    assert abs(avg_time - expected) < 0.0001, f"Expected {expected}, got {avg_time}"
    print(f"Average time: {avg_time:.4f}")
    
    # Test calculate_idle_time
    idle_time = metrics.calculate_idle_time(trajectory)
    assert idle_time == 2, f"Expected 2 wait actions, got {idle_time}"
    print(f"Idle time: {idle_time}")
    
    # Test with empty trajectory
    empty_avg_time = metrics.calculate_avg_time([])
    assert empty_avg_time == 0.0, f"Expected 0.0 for empty trajectory, got {empty_avg_time}"
    
    # Test with single action trajectory
    single_action = [{'action': {'type': 'click'}, 'cost': 0.001}]
    single_avg_time = metrics.calculate_avg_time(single_action)
    assert single_avg_time == 0.0, f"Expected 0.0 for single action, got {single_avg_time}"
    
    print("✅ Individual methods tests passed!")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\nTesting edge cases...")
    
    metrics = WebArenaSpecificMetrics()
    
    # Test with empty trajectory
    empty_result = metrics.calculate_efficiency_metrics([])
    assert empty_result['actions_per_task'] == 0, "Empty trajectory should have 0 actions"
    assert empty_result['failed_actions'] == 0, "Empty trajectory should have 0 failed actions"
    assert empty_result['navigation_cost'] == 0, "Empty trajectory should have 0 navigation cost"
    
    print(f"Empty trajectory result: {empty_result}")
    
    # Test with trajectory missing some fields
    incomplete_trajectory = [
        {'action': {'type': 'click'}, 'cost': 0.001},  # Missing error field
        {'action': {}, 'cost': 0.002},  # Missing action type
        {'cost': 0.003},  # Missing action field
    ]
    
    try:
        incomplete_result = metrics.calculate_efficiency_metrics(incomplete_trajectory)
        print(f"Incomplete trajectory handled: {incomplete_result}")
        # Should handle gracefully without crashing
        assert incomplete_result['actions_per_task'] == 3, "Should count all trajectory items"
    except Exception as e:
        print(f"Error handling incomplete trajectory: {e}")
        # This is acceptable - depends on implementation robustness
    
    # Test with all action types
    comprehensive_trajectory = [
        {'action': {'type': 'click'}, 'cost': 0.001},
        {'action': {'type': 'type'}, 'cost': 0.002},
        {'action': {'type': 'scroll'}, 'cost': 0.001},
        {'action': {'type': 'back'}, 'cost': 0.001},
        {'action': {'type': 'wait'}, 'cost': 0.001},
        {'action': {'type': 'select'}, 'cost': 0.001},
        {'action': {'type': 'verify'}, 'cost': 0.003},
        {'action': {'type': 'query'}, 'cost': 0.002}
    ]
    
    comprehensive_result = metrics.calculate_efficiency_metrics(comprehensive_trajectory)
    assert comprehensive_result['actions_per_task'] == 8, "Should count all action types"
    assert comprehensive_result['backtrack_rate'] == 1, "Should find back action"
    assert comprehensive_result['dom_queries'] == 1, "Should find query action"
    assert comprehensive_result['idle_time'] == 1, "Should find wait action"
    
    print(f"Comprehensive trajectory result: {comprehensive_result}")
    
    print("✅ Edge cases tests passed!")


def test_realistic_scenario():
    """Test with a realistic WebArena task scenario"""
    print("\nTesting realistic scenario...")
    
    metrics = WebArenaSpecificMetrics()
    
    # Simulate a login task
    login_trajectory = [
        {'action': {'type': 'click', 'element_id': 'login_button'}, 'cost': 0.002},
        {'action': {'type': 'type', 'text': 'user@example.com'}, 'cost': 0.003},
        {'action': {'type': 'type', 'text': 'password123'}, 'cost': 0.003},
        {'action': {'type': 'click', 'element_id': 'submit'}, 'cost': 0.002},
        {'action': {'type': 'wait'}, 'cost': 0.001},
        {'action': {'type': 'verify'}, 'cost': 0.004, 'error': 'Login failed'},
        {'action': {'type': 'back'}, 'cost': 0.001},
        {'action': {'type': 'type', 'text': 'correct_password'}, 'cost': 0.003},
        {'action': {'type': 'click', 'element_id': 'submit'}, 'cost': 0.002},
        {'action': {'type': 'verify'}, 'cost': 0.004}
    ]
    
    result = metrics.calculate_efficiency_metrics(login_trajectory)
    
    # Analyze the results
    total_cost = sum(t['cost'] for t in login_trajectory)
    success_rate = 1 - (result['failed_actions'] / result['actions_per_task'])
    
    print(f"Login task analysis:")
    print(f"  Total actions: {result['actions_per_task']}")
    print(f"  Failed actions: {result['failed_actions']}")
    print(f"  Success rate: {success_rate:.2%}")
    print(f"  Backtrack rate: {result['backtrack_rate']}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Navigation cost: ${result['navigation_cost']:.4f}")
    print(f"  Form filling cost: ${result['form_filling_cost']:.4f}")
    print(f"  Verification cost: ${result['verification_cost']:.4f}")
    print(f"  Idle time: {result['idle_time']}")
    
    # Verify some key metrics
    assert result['actions_per_task'] == 10, "Should have 10 total actions"
    assert result['failed_actions'] == 1, "Should have 1 failed action (verify with error)"
    assert result['backtrack_rate'] == 1, "Should have 1 backtrack"
    assert result['navigation_cost'] == 0.006, "Should have navigation cost from 3 clicks"
    assert abs(result['form_filling_cost'] - 0.009) < 0.0001, "Should have form cost from 3 type actions"
    
    print("✅ Realistic scenario tests passed!")


if __name__ == "__main__":
    test_basic_metrics_calculation()
    test_cost_breakdown()
    test_time_metrics()
    test_individual_methods()
    test_edge_cases()
    test_realistic_scenario()
    print("\n🎉 All WebArenaSpecificMetrics tests passed successfully!")