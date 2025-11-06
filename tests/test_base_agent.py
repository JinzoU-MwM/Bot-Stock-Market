import pytest
from agents.base_agent import BaseAgent
from agents.agent_factory import AgentFactory


def test_create_conservative_agent():
    agent = AgentFactory.create_agent("conservative", "agent_1", 500000000)
    assert agent.name == "agent_1"
    assert agent.strategy_type == "conservative"
    assert agent.initial_capital == 500000000
    assert agent.current_balance == 500000000


def test_agent_state_representation():
    agent = AgentFactory.create_agent("balanced", "agent_2", 500000000)
    state = agent.get_state("AAPL")

    required_keys = [
        "cash_balance",
        "positions",
        "total_pnl",
        "rsi",
        "macd",
        "market_volatility",
    ]
    for key in required_keys:
        assert key in state


def test_agent_action_execution():
    agent = AgentFactory.create_agent("aggressive", "agent_3", 500000000)
    action = {"type": "BUY", "symbol": "AAPL", "quantity": 100, "price": 150}

    result = agent.execute_action(action)
    assert result["success"] == True
    assert agent.current_balance < 500000000
