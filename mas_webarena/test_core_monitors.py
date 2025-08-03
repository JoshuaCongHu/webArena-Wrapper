#!/usr/bin/env python3
"""Test the Core Monitor Classes as specified in CLAUDE.md"""

import time
from monitors.communication_monitor import CommunicationMonitor
from monitors.action_monitor import ActionMonitor


def test_communication_monitor():
    """Test CommunicationMonitor functionality"""
    print("Testing CommunicationMonitor...")
    
    monitor = CommunicationMonitor()
    
    # Test logging messages
    monitor.log_message("agent_1", "agent_2", "Can you help with task X?", "request")
    monitor.log_message("agent_2", "agent_1", "Yes, I can handle that", "response")
    monitor.log_message("orchestrator", "agent_1", "Status update required", "status")
    
    # Test step messages
    step_messages = monitor.get_step_messages()
    print(f"Step messages: {len(step_messages)}")
    assert len(step_messages) == 3, "Should have 3 step messages"
    
    # Test that step messages buffer is cleared
    step_messages_2 = monitor.get_step_messages()
    assert len(step_messages_2) == 0, "Step messages buffer should be cleared"
    
    # Test stats
    stats = monitor.get_stats()
    print(f"Total messages: {stats['total_messages']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Messages by type: {dict(stats['messages_by_type'])}")
    print(f"Messages by sender: {dict(stats['messages_by_sender'])}")
    
    assert stats['total_messages'] == 3, "Should have 3 total messages"
    assert stats['messages_by_type']['request'] == 1, "Should have 1 request message"
    assert stats['messages_by_sender']['agent_1'] == 1, "Agent_1 should have sent 1 message"
    
    # Test reset
    monitor.reset()
    stats_after_reset = monitor.get_stats()
    assert stats_after_reset['total_messages'] == 0, "Should have 0 messages after reset"
    
    print("✅ CommunicationMonitor tests passed!")


def test_action_monitor():
    """Test ActionMonitor functionality"""
    print("\nTesting ActionMonitor...")
    
    monitor = ActionMonitor()
    
    # Test logging actions
    monitor.log_action({'type': 'click', 'element_id': 123}, 0.001)
    monitor.log_action({'type': 'type', 'text': 'hello world'}, 0.002)
    monitor.log_action({'type': 'scroll', 'direction': 'down'}, 0.0005)
    monitor.log_action({'type': 'click', 'element_id': 456}, 0.001)
    monitor.log_action({'type': 'wait'}, 0.0001)
    
    # Test stats
    stats = monitor.get_stats()
    print(f"Total actions: {stats['total_actions']}")
    print(f"Total cost: ${stats['total_cost']:.4f}")
    print(f"Action counts: {stats['action_counts']}")
    print(f"Avg cost per action: ${stats['avg_cost_per_action']:.4f}")
    
    assert stats['total_actions'] == 5, "Should have 5 total actions"
    assert stats['action_counts']['click'] == 2, "Should have 2 click actions"
    assert stats['action_counts']['type'] == 1, "Should have 1 type action"
    assert abs(stats['total_cost'] - 0.0046) < 0.0001, "Total cost should be 0.0046"
    
    # Test reset
    monitor.reset()
    stats_after_reset = monitor.get_stats()
    assert stats_after_reset['total_actions'] == 0, "Should have 0 actions after reset"
    assert stats_after_reset['total_cost'] == 0, "Should have 0 cost after reset"
    
    print("✅ ActionMonitor tests passed!")


def test_monitor_integration():
    """Test monitors working together"""
    print("\nTesting monitor integration...")
    
    comm_monitor = CommunicationMonitor()
    action_monitor = ActionMonitor()
    
    # Simulate a task execution scenario
    comm_monitor.log_message("orchestrator", "agent_1", "Execute click action", "instruction")
    action_monitor.log_action({'type': 'click', 'element_id': 789}, 0.001)
    
    comm_monitor.log_message("agent_1", "orchestrator", "Click action completed", "report")
    
    comm_monitor.log_message("orchestrator", "agent_2", "Verify result", "instruction")
    action_monitor.log_action({'type': 'wait'}, 0.0001)
    
    # Get combined stats
    comm_stats = comm_monitor.get_stats()
    action_stats = action_monitor.get_stats()
    
    print(f"Communication: {comm_stats['total_messages']} messages, {comm_stats['total_tokens']} tokens")
    print(f"Actions: {action_stats['total_actions']} actions, ${action_stats['total_cost']:.4f} cost")
    
    assert comm_stats['total_messages'] == 3, "Should have 3 communication messages"
    assert action_stats['total_actions'] == 2, "Should have 2 actions"
    
    print("✅ Monitor integration tests passed!")


if __name__ == "__main__":
    test_communication_monitor()
    test_action_monitor()
    test_monitor_integration()
    print("\n🎉 All Core Monitor tests passed successfully!")