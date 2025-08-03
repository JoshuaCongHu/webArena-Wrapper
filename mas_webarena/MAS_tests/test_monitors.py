#!/usr/bin/env python3
"""Simple integration test for the monitor classes"""

import time
from WebArenaMetrics import WebArenaMetrics
from webArenaCommunication import WebArenaCommunicationMonitor, MessageType
from webArenaCostMonitor import WebArenaCostMonitor


def test_monitor_integration():
    """Test that all monitor classes work together"""
    print("Testing WebArena Monitor Classes Integration...")
    
    # Initialize monitors
    metrics = WebArenaMetrics()
    comm_monitor = WebArenaCommunicationMonitor()
    cost_monitor = WebArenaCostMonitor()
    
    # Simulate some agent activities
    print("\n1. Testing WebArenaMetrics...")
    
    # Track some actions
    metrics.track_action("agent_1", "click", True, 0.5, 0.001)
    metrics.track_action("agent_1", "type", True, 1.2, 0.002)
    metrics.track_action("agent_2", "navigate", False, 2.1, 0.003, "Navigation failed")
    metrics.track_action("agent_2", "verify", True, 0.8, 0.002)
    
    efficiency = metrics.get_efficiency_metrics()
    print(f"Success rate: {efficiency['success_rate']:.2%}")
    print(f"Average execution time: {efficiency['average_execution_time']:.2f}s")
    print(f"Total cost: ${efficiency['total_cost']:.4f}")
    
    # Test agent performance
    agent1_perf = metrics.get_agent_performance("agent_1")
    print(f"Agent 1 success rate: {agent1_perf['success_rate']:.2%}")
    
    print("\n2. Testing WebArenaCommunicationMonitor...")
    
    # Log some communications
    comm_monitor.log_message("agent_1", "agent_2", MessageType.TASK_REQUEST, 
                            {"task": "click_button", "element_id": "submit"}, 0.1)
    comm_monitor.log_message("agent_2", "agent_1", MessageType.TASK_RESPONSE, 
                            {"status": "completed", "result": "success"}, 0.05)
    comm_monitor.log_message("agent_1", "orchestrator", MessageType.STATUS_UPDATE, 
                            {"progress": 0.5, "current_task": "form_filling"}, 0.02)
    
    comm_efficiency = comm_monitor.get_communication_efficiency()
    print(f"Total messages: {comm_efficiency['total_messages']}")
    print(f"Average response time: {comm_efficiency['average_response_time']:.3f}s")
    
    # Test communication profile
    agent1_comm = comm_monitor.get_agent_communication_profile("agent_1")
    print(f"Agent 1 messages sent: {agent1_comm['messages_sent']}")
    print(f"Agent 1 messages received: {agent1_comm['messages_received']}")
    
    print("\n3. Testing WebArenaCostMonitor...")
    
    # Track some costs
    cost_monitor.track_llm_call("gpt-4", 150, 75)
    cost_monitor.track_vision_analysis(2048 * 1024)  # 2MB image
    cost_monitor.track_action_execution("click", 1.5)
    cost_monitor.track_action_execution("type", 2.0)
    
    cost_breakdown = cost_monitor.get_cost_breakdown()
    print(f"Total cost: ${cost_breakdown['total_cost']:.4f}")
    print(f"LLM calls: ${cost_breakdown['llm_calls']:.4f}")
    print(f"Vision API: ${cost_breakdown['vision_api_calls']:.4f}")
    print(f"Actions: ${cost_breakdown['action_executions']:.4f}")
    
    cost_efficiency = cost_monitor.get_cost_efficiency_metrics()
    print(f"Cost per token: ${cost_efficiency['cost_per_token']:.6f}")
    
    print("\n4. Testing Combined Metrics...")
    
    # Get comprehensive summary
    task_summary = metrics.get_task_summary()
    network_analysis = comm_monitor.get_communication_network_analysis()
    
    print(f"Session duration: {task_summary['session_info']['duration_seconds']:.2f}s")
    print(f"Total agents in network: {network_analysis['total_agents']}")
    print(f"Network density: {network_analysis['network_density']:.2f}")
    
    # Check for anomalies
    anomalies = comm_monitor.detect_communication_anomalies()
    if anomalies:
        print(f"\nDetected {len(anomalies)} communication anomalies:")
        for anomaly in anomalies:
            print(f"- {anomaly['type']}: {anomaly['description']}")
    else:
        print("\nNo communication anomalies detected.")
    
    print("\n✅ All monitor classes working correctly!")
    return True


if __name__ == "__main__":
    test_monitor_integration()