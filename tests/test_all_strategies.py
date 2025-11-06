import pytest
from agents.agent_factory import AgentFactory


def test_all_strategy_types():
    """Test that all strategy types can be created"""
    strategies = ["conservative", "aggressive", "balanced", "trend", "mean_reversion"]

    for strategy in strategies:
        agent = AgentFactory.create_agent(strategy, f"test_{strategy}", 1000000)

        # Test basic properties
        assert agent.strategy_type == strategy
        assert agent.name == f"test_{strategy}"
        assert agent.initial_capital == 1000000
        assert agent.current_balance == 1000000

        # Test strategy-specific methods
        max_position = agent.get_max_position_size()
        risk_tolerance = agent.get_risk_tolerance()

        assert isinstance(max_position, float)
        assert 0 < max_position <= 0.25  # Should be reasonable range
        assert risk_tolerance in ["low", "medium", "high"]


def test_agent_position_limits():
    """Test that agents respect position limits"""
    agent = AgentFactory.create_agent("conservative", "position_test", 500000000)

    # Test max 3 positions limit
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

    for i, symbol in enumerate(symbols):
        action = {"type": "BUY", "symbol": symbol, "quantity": 100, "price": 150}
        result = agent.execute_action(action)

        if i < 3:  # First 3 should succeed
            assert result["success"] == True
        else:  # 4th should fail due to position limit
            assert result["success"] == False
            assert "Maximum positions reached" in result["error"]


def test_agent_insufficient_funds():
    """Test that agents can't spend more than they have"""
    agent = AgentFactory.create_agent("balanced", "funds_test", 1000)  # Small capital

    # Try to buy more than can afford
    action = {"type": "BUY", "symbol": "AAPL", "quantity": 1000, "price": 150}
    result = agent.execute_action(action)

    assert result["success"] == False
    assert "Insufficient funds" in result["error"]


def test_agent_sell_without_position():
    """Test that agents can't sell positions they don't have"""
    agent = AgentFactory.create_agent("aggressive", "sell_test", 500000000)

    # Try to sell without having position
    action = {"type": "SELL", "symbol": "AAPL", "quantity": 100, "price": 150}
    result = agent.execute_action(action)

    assert result["success"] == False
    assert "No position to sell" in result["error"]


def test_agent_buy_sell_cycle():
    """Test complete buy and sell cycle"""
    agent = AgentFactory.create_agent("balanced", "cycle_test", 500000000)

    initial_balance = agent.current_balance

    # Buy a position
    buy_action = {"type": "BUY", "symbol": "AAPL", "quantity": 100, "price": 150}
    buy_result = agent.execute_action(buy_action)

    assert buy_result["success"] == True
    assert agent.current_balance < initial_balance
    assert "AAPL" in agent.positions

    # Sell the position at profit
    sell_action = {"type": "SELL", "symbol": "AAPL", "quantity": 100, "price": 160}
    sell_result = agent.execute_action(sell_action)

    assert sell_result["success"] == True
    assert "AAPL" not in agent.positions
    assert (
        agent.current_balance > initial_balance
    )  # Should be higher after profitable trade
    assert sell_result["profit_loss"] > 0
