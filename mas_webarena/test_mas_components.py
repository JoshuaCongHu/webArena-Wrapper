#!/usr/bin/env python3
"""Test the MAS System Components as specified in CLAUDE.md"""

import time
from utils.budget_tracker import BudgetTracker
from utils.mas_response import MASResponse


def test_budget_tracker():
    """Test BudgetTracker functionality"""
    print("Testing BudgetTracker...")
    
    # Test initialization
    tracker = BudgetTracker(initial_budget=1.0)
    assert tracker.initial_budget == 1.0, "Should initialize with correct budget"
    assert tracker.remaining == 1.0, "Should start with full budget remaining"
    assert tracker.spent == 0.0, "Should start with no spending"
    assert len(tracker.history) == 0, "Should start with empty history"
    
    print(f"Initial budget: {tracker.initial_budget}")
    print(f"Remaining: {tracker.remaining}")
    print(f"Spent: {tracker.spent}")
    
    # Test successful consumption
    success1 = tracker.consume(0.3)
    assert success1 == True, "Should successfully consume available budget"
    assert abs(tracker.remaining - 0.7) < 0.0001, "Should have 0.7 remaining"
    assert abs(tracker.spent - 0.3) < 0.0001, "Should have spent 0.3"
    assert len(tracker.history) == 1, "Should record transaction in history"
    
    print(f"After consuming 0.3 - Remaining: {tracker.remaining}, Spent: {tracker.spent}")
    
    # Test another consumption
    success2 = tracker.consume(0.5)
    assert success2 == True, "Should successfully consume more budget"
    assert abs(tracker.remaining - 0.2) < 0.0001, "Should have 0.2 remaining"
    assert abs(tracker.spent - 0.8) < 0.0001, "Should have spent 0.8"
    assert len(tracker.history) == 2, "Should have 2 transactions in history"
    
    print(f"After consuming 0.5 - Remaining: {tracker.remaining}, Spent: {tracker.spent}")
    
    # Test insufficient budget
    success3 = tracker.consume(0.5)  # Trying to consume more than remaining
    assert success3 == False, "Should fail when insufficient budget"
    assert abs(tracker.remaining - 0.2) < 0.0001, "Remaining should be unchanged"
    assert abs(tracker.spent - 0.8) < 0.0001, "Spent should be unchanged"
    assert len(tracker.history) == 2, "History should be unchanged"
    
    print(f"After failed consumption - Remaining: {tracker.remaining}, Spent: {tracker.spent}")
    
    # Test history structure
    first_transaction = tracker.history[0]
    assert 'amount' in first_transaction, "History should record amount"
    assert 'remaining' in first_transaction, "History should record remaining"
    assert 'timestamp' in first_transaction, "History should record timestamp"
    assert first_transaction['amount'] == 0.3, "Should record correct amount"
    
    print(f"First transaction: {first_transaction}")
    
    # Test reset
    tracker.reset()
    assert tracker.remaining == 1.0, "Should reset to initial budget"
    assert tracker.spent == 0.0, "Should reset spent to 0"
    assert len(tracker.history) == 0, "Should clear history"
    
    print(f"After reset - Remaining: {tracker.remaining}, Spent: {tracker.spent}")
    
    # Test reset with new budget
    tracker.reset(budget=2.0)
    assert tracker.initial_budget == 2.0, "Should update initial budget"
    assert tracker.remaining == 2.0, "Should reset to new budget"
    
    print(f"After reset with new budget - Initial: {tracker.initial_budget}, Remaining: {tracker.remaining}")
    
    print("✅ BudgetTracker tests passed!")


def test_mas_response():
    """Test MASResponse functionality"""
    print("\nTesting MASResponse...")
    
    # Test creation with required fields
    response = MASResponse(
        agent_id="test_agent_1",
        action={'type': 'click', 'element_id': 123},
        reasoning="Found a clickable button element",
        confidence=0.85
    )
    
    assert response.agent_id == "test_agent_1", "Should set agent_id correctly"
    assert response.action['type'] == 'click', "Should set action correctly"
    assert response.reasoning == "Found a clickable button element", "Should set reasoning correctly"
    assert response.confidence == 0.85, "Should set confidence correctly"
    assert response.tokens_used == 0, "Should default tokens_used to 0"
    assert response.used_vision == False, "Should default used_vision to False"
    
    print(f"Basic response: {response}")
    
    # Test creation with all fields
    full_response = MASResponse(
        agent_id="vision_agent",
        action={'type': 'type', 'text': 'hello world'},
        reasoning="Need to fill out the form field",
        confidence=0.95,
        tokens_used=150,
        used_vision=True
    )
    
    assert full_response.tokens_used == 150, "Should set tokens_used correctly"
    assert full_response.used_vision == True, "Should set used_vision correctly"
    
    print(f"Full response: {full_response}")
    
    # Test to_dict() method
    response_dict = response.to_dict()
    expected_keys = {'agent_id', 'action', 'reasoning', 'confidence', 'tokens_used', 'used_vision'}
    assert set(response_dict.keys()) == expected_keys, "Should include all expected keys"
    assert response_dict['agent_id'] == "test_agent_1", "Dict should preserve agent_id"
    assert response_dict['action']['type'] == 'click', "Dict should preserve action"
    assert response_dict['confidence'] == 0.85, "Dict should preserve confidence"
    
    print(f"Response dict: {response_dict}")
    
    # Test with complex action
    complex_response = MASResponse(
        agent_id="form_agent",
        action={
            'type': 'type',
            'element_id': 456,
            'text': 'user@example.com',
            'clear_first': True
        },
        reasoning="Filling email field with validation",
        confidence=0.92,
        tokens_used=75
    )
    
    complex_dict = complex_response.to_dict()
    assert complex_dict['action']['element_id'] == 456, "Should preserve complex action structure"
    assert complex_dict['action']['clear_first'] == True, "Should preserve boolean values in action"
    
    print(f"Complex response dict: {complex_dict}")
    
    print("✅ MASResponse tests passed!")


def test_components_integration():
    """Test BudgetTracker and MASResponse working together"""
    print("\nTesting component integration...")
    
    # Simulate a task execution scenario
    tracker = BudgetTracker(initial_budget=0.5)
    
    # First agent action
    response1 = MASResponse(
        agent_id="navigator",
        action={'type': 'click', 'element_id': 1},
        reasoning="Clicking navigation link",
        confidence=0.9,
        tokens_used=50
    )
    
    # Estimate cost based on response
    cost1 = 0.001 + (response1.tokens_used * 0.00003)  # Base cost + token cost
    success1 = tracker.consume(cost1)
    
    assert success1 == True, "Should successfully consume budget for first action"
    print(f"Action 1 cost: ${cost1:.5f}, Remaining budget: ${tracker.remaining:.5f}")
    
    # Second agent action with vision
    response2 = MASResponse(
        agent_id="vision_agent",
        action={'type': 'type', 'text': 'search term'},
        reasoning="Filling search field based on screenshot analysis",
        confidence=0.8,
        tokens_used=100,
        used_vision=True
    )
    
    # Higher cost due to vision usage
    cost2 = 0.001 + (response2.tokens_used * 0.00003) + (0.01 if response2.used_vision else 0)
    success2 = tracker.consume(cost2)
    
    assert success2 == True, "Should successfully consume budget for vision action"
    print(f"Action 2 cost: ${cost2:.5f}, Remaining budget: ${tracker.remaining:.5f}")
    
    # Third action that exceeds budget
    response3 = MASResponse(
        agent_id="heavy_agent",
        action={'type': 'verify'},
        reasoning="Complex verification task",
        confidence=0.7,
        tokens_used=50000  # Very high token usage to exceed remaining budget
    )
    
    cost3 = 0.001 + (response3.tokens_used * 0.00003)
    success3 = tracker.consume(cost3)
    
    assert success3 == False, "Should fail when budget exceeded"
    print(f"Action 3 cost: ${cost3:.5f}, Budget exceeded - Remaining: ${tracker.remaining:.5f}")
    
    # Verify total spending
    total_expected = cost1 + cost2
    assert abs(tracker.spent - total_expected) < 0.00001, "Should track total spending correctly"
    
    print(f"Total spent: ${tracker.spent:.5f}, Expected: ${total_expected:.5f}")
    print(f"Budget utilization: {(tracker.spent / tracker.initial_budget) * 100:.1f}%")
    
    print("✅ Component integration tests passed!")


if __name__ == "__main__":
    test_budget_tracker()
    test_mas_response()
    test_components_integration()
    print("\n🎉 All MAS System Component tests passed successfully!")