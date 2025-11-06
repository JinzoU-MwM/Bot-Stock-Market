import pytest
from services.coordinator import Coordinator
from agents.agent_factory import AgentFactory


def test_coordinator_initialization():
    coordinator = Coordinator()
    assert len(coordinator.agents) == 0
    assert coordinator.rl_engine is not None


def test_add_agents():
    coordinator = Coordinator()

    strategies = ["conservative", "aggressive", "balanced", "trend", "mean_reversion"]
    for i, strategy in enumerate(strategies):
        agent = AgentFactory.create_agent(strategy, f"agent_{i}", 500000000)
        coordinator.add_agent(agent)

    assert len(coordinator.agents) == 5


def test_start_simulation():
    coordinator = Coordinator()
    agent = AgentFactory.create_agent("balanced", "test_agent", 500000000)
    coordinator.add_agent(agent)

    # Start simulation for a few steps
    results = coordinator.run_simulation_steps(5)
    assert len(results) == 5
    # Check that the last step has performance metrics for the agent
    assert agent.id in results[-1]
    assert "performance" in results[-1][agent.id]
    assert "total_pnl" in results[-1][agent.id]["performance"]
