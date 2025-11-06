import pytest
import uuid
from datetime import datetime
from agents.base_agent import BaseAgent
from agents.agent_factory import AgentFactory


class TestAgentFactory:
    """Test the Agent Factory pattern"""

    def test_create_all_strategy_types(self):
        """Test creating all available strategy types"""
        strategies = AgentFactory.get_available_strategies()
        expected_strategies = [
            "conservative",
            "aggressive",
            "balanced",
            "trend",
            "mean_reversion",
        ]

        assert strategies == expected_strategies

        for strategy in strategies:
            agent = AgentFactory.create_agent(strategy, f"test_{strategy}", 1000000)
            assert agent.strategy_type == strategy
            assert agent.name == f"test_{strategy}"
            assert agent.initial_capital == 1000000

    def test_create_invalid_strategy(self):
        """Test creating agent with invalid strategy type"""
        with pytest.raises(ValueError, match="Unknown strategy type: invalid"):
            AgentFactory.create_agent("invalid", "test_agent", 1000000)

    def test_agent_uniqueness(self):
        """Test that each agent gets unique UUID"""
        agent1 = AgentFactory.create_agent("balanced", "agent1", 1000000)
        agent2 = AgentFactory.create_agent("balanced", "agent2", 1000000)

        assert agent1.id != agent2.id
        assert agent1.episode_id != agent2.episode_id


class TestBaseAgent:
    """Test the Base Agent functionality"""

    def test_agent_initialization(self):
        """Test proper agent initialization"""
        agent = AgentFactory.create_agent("conservative", "test_agent", 500000000)

        assert agent.name == "test_agent"
        assert agent.strategy_type == "conservative"
        assert agent.initial_capital == 500000000
        assert agent.current_balance == 500000000
        assert agent.status == "active"
        assert agent.positions == {}
        assert agent.trades == []
        assert agent.step_count == 0
        assert isinstance(agent.id, str)
        assert isinstance(agent.episode_id, str)

    def test_state_representation_completeness(self):
        """Test that state representation includes all required fields"""
        agent = AgentFactory.create_agent("balanced", "test_agent", 500000000)

        # Test with no market data
        state = agent.get_state("AAPL")

        required_fields = [
            "cash_balance",
            "positions",
            "total_pnl",
            "available_margin",
            "portfolio_value",
            "rsi",
            "macd",
            "bollinger_position",
            "atr",
            "volume_ratio",
            "market_volatility",
            "sentiment_score",
            "hour_of_day",
            "day_of_week",
            "market_session",
        ]

        for field in required_fields:
            assert field in state, f"Missing field: {field}"

        # Test with market data
        market_data = {
            "rsi": 65.5,
            "macd": 1.2,
            "atr": 2.5,
            "volume_ratio": 1.5,
            "price": 150.0,
            "bollinger_upper": 155.0,
            "bollinger_lower": 145.0,
        }

        state_with_data = agent.get_state("AAPL", market_data)
        assert state_with_data["rsi"] == 65.5
        assert state_with_data["macd"] == 1.2
        assert state_with_data["bollinger_position"] > 0  # Should be calculated

    def test_market_session_detection(self):
        """Test market session detection logic"""
        agent = AgentFactory.create_agent("balanced", "test_agent", 500000000)

        # Mock different hours
        original_hour = datetime.now().hour

        # Test US market session (9 AM - 4 PM)
        for hour in range(9, 17):
            agent._test_hour = hour
            session = agent.get_market_session()
            assert session == "us", f"Hour {hour} should be US session"

        # Test Asian market session (before 9 AM)
        for hour in range(0, 9):
            agent._test_hour = hour
            session = agent.get_market_session()
            assert session == "asia", f"Hour {hour} should be Asia session"

        # Test European market session (after 4 PM)
        for hour in range(17, 24):
            agent._test_hour = hour
            session = agent.get_market_session()
            assert session == "europe", f"Hour {hour} should be Europe session"

    def test_buy_order_execution_success(self):
        """Test successful buy order execution"""
        agent = AgentFactory.create_agent("aggressive", "test_agent", 500000000)
        action = {"type": "BUY", "symbol": "AAPL", "quantity": 100, "price": 150.0}

        result = agent.execute_action(action)

        assert result["success"] is True
        assert "trade" in result
        assert result["new_balance"] < 500000000
        assert result["positions_count"] == 1

        # Verify position was created
        assert "AAPL" in agent.positions
        assert agent.positions["AAPL"]["quantity"] == 100
        assert agent.positions["AAPL"]["entry_price"] == 150.0

        # Verify trade was recorded
        assert len(agent.trades) == 1
        trade = agent.trades[0]
        assert trade["type"] == "BUY"
        assert trade["symbol"] == "AAPL"
        assert trade["quantity"] == 100
        assert trade["price"] == 150.0
        assert trade["commission"] > 0

    def test_buy_order_insufficient_funds(self):
        """Test buy order with insufficient funds"""
        agent = AgentFactory.create_agent(
            "conservative", "test_agent", 1000
        )  # Low capital
        action = {
            "type": "BUY",
            "symbol": "AAPL",
            "quantity": 100,
            "price": 150.0,  # Total cost would be > 1000
        }

        result = agent.execute_action(action)

        assert result["success"] is False
        assert "Insufficient funds" in result["error"]
        assert len(agent.positions) == 0
        assert len(agent.trades) == 0

    def test_buy_order_position_already_exists(self):
        """Test buy order when position already exists"""
        agent = AgentFactory.create_agent("balanced", "test_agent", 500000000)

        # Create initial position
        action1 = {"type": "BUY", "symbol": "AAPL", "quantity": 100, "price": 150.0}
        agent.execute_action(action1)

        # Try to buy same symbol again
        action2 = {"type": "BUY", "symbol": "AAPL", "quantity": 50, "price": 155.0}
        result = agent.execute_action(action2)

        assert result["success"] is False
        assert "Position already exists" in result["error"]
        assert agent.positions["AAPL"]["quantity"] == 100  # Unchanged

    def test_buy_order_max_positions_reached(self):
        """Test buy order when maximum positions reached"""
        agent = AgentFactory.create_agent("aggressive", "test_agent", 500000000)

        # Create 3 positions (maximum allowed)
        symbols = ["AAPL", "GOOGL", "MSFT"]
        for symbol in symbols:
            action = {"type": "BUY", "symbol": symbol, "quantity": 10, "price": 100.0}
            agent.execute_action(action)

        # Try to create 4th position
        action = {"type": "BUY", "symbol": "TSLA", "quantity": 10, "price": 100.0}
        result = agent.execute_action(action)

        assert result["success"] is False
        assert "Maximum positions reached" in result["error"]
        assert len(agent.positions) == 3

    def test_sell_order_success(self):
        """Test successful sell order"""
        agent = AgentFactory.create_agent("balanced", "test_agent", 500000000)

        # First buy a position
        buy_action = {"type": "BUY", "symbol": "AAPL", "quantity": 100, "price": 150.0}
        agent.execute_action(buy_action)

        # Then sell it
        sell_action = {
            "type": "SELL",
            "symbol": "AAPL",
            "quantity": 100,
            "price": 160.0,
        }
        result = agent.execute_action(sell_action)

        assert result["success"] is True
        assert "trade" in result
        assert "profit_loss" in result
        assert result["profit_loss"] > 0  # Should be profitable
        assert (
            result["new_balance"] > agent.current_balance - 16000
        )  # Balance increased
        assert result["positions_count"] == 0  # Position closed

        # Verify position was removed
        assert "AAPL" not in agent.positions

        # Verify sell trade was recorded
        assert len(agent.trades) == 2
        sell_trade = agent.trades[1]
        assert sell_trade["type"] == "SELL"
        assert sell_trade["profit_loss"] > 0

    def test_sell_order_no_position(self):
        """Test sell order when no position exists"""
        agent = AgentFactory.create_agent("balanced", "test_agent", 500000000)

        action = {"type": "SELL", "symbol": "AAPL", "quantity": 100, "price": 150.0}
        result = agent.execute_action(action)

        assert result["success"] is False
        assert "No position to sell" in result["error"]
        assert len(agent.trades) == 0

    def test_sell_order_insufficient_position(self):
        """Test sell order with insufficient position quantity"""
        agent = AgentFactory.create_agent("balanced", "test_agent", 500000000)

        # Buy small position
        buy_action = {"type": "BUY", "symbol": "AAPL", "quantity": 50, "price": 150.0}
        agent.execute_action(buy_action)

        # Try to sell more than we have
        sell_action = {
            "type": "SELL",
            "symbol": "AAPL",
            "quantity": 100,
            "price": 160.0,
        }
        result = agent.execute_action(sell_action)

        assert result["success"] is False
        assert "Insufficient position quantity" in result["error"]
        assert agent.positions["AAPL"]["quantity"] == 50  # Unchanged

    def test_commission_calculation(self):
        """Test that commission is properly calculated (0.1%)"""
        agent = AgentFactory.create_agent("balanced", "test_agent", 500000000)

        action = {"type": "BUY", "symbol": "AAPL", "quantity": 100, "price": 150.0}
        result = agent.execute_action(action)

        expected_commission = 100 * 150.0 * 0.001  # 15.0
        assert result["trade"]["commission"] == expected_commission
        assert agent.current_balance == 500000000 - 15000 - expected_commission

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        agent = AgentFactory.create_agent("balanced", "test_agent", 500000000)

        # Initially no trades
        metrics = agent.get_performance_metrics()
        assert metrics["total_trades"] == 0
        assert metrics["winning_trades"] == 0
        assert metrics["losing_trades"] == 0
        assert metrics["win_rate"] == 0
        assert metrics["total_pnl"] == 0
        assert metrics["total_return"] == 0

        # Execute some trades
        # Buy AAPL at 150, sell at 160 (profit)
        agent.execute_action(
            {"type": "BUY", "symbol": "AAPL", "quantity": 100, "price": 150.0}
        )
        agent.execute_action(
            {"type": "SELL", "symbol": "AAPL", "quantity": 100, "price": 160.0}
        )

        # Buy GOOGL at 100, sell at 90 (loss)
        agent.execute_action(
            {"type": "BUY", "symbol": "GOOGL", "quantity": 100, "price": 100.0}
        )
        agent.execute_action(
            {"type": "SELL", "symbol": "GOOGL", "quantity": 100, "price": 90.0}
        )

        metrics = agent.get_performance_metrics()
        assert metrics["total_trades"] == 2
        assert metrics["winning_trades"] == 1
        assert metrics["losing_trades"] == 1
        assert metrics["win_rate"] == 0.5
        assert (
            metrics["total_pnl"] > -2000 and metrics["total_pnl"] < 2000
        )  # Rough range
        assert metrics["total_return"] != 0


class TestStrategyAgents:
    """Test different strategy implementations"""

    def test_conservative_strategy(self):
        """Test Conservative agent properties"""
        agent = AgentFactory.create_agent("conservative", "test_agent", 500000000)

        assert agent.get_max_position_size() == 0.10  # 10% position size
        assert agent.get_risk_tolerance() == "low"

    def test_aggressive_strategy(self):
        """Test Aggressive agent properties"""
        agent = AgentFactory.create_agent("aggressive", "test_agent", 500000000)

        assert agent.get_max_position_size() == 0.20  # 20% position size
        assert agent.get_risk_tolerance() == "high"

    def test_balanced_strategy(self):
        """Test Balanced agent properties"""
        agent = AgentFactory.create_agent("balanced", "test_agent", 500000000)

        assert agent.get_max_position_size() == 0.15  # 15% position size
        assert agent.get_risk_tolerance() == "medium"

    def test_trend_strategy(self):
        """Test Trend agent properties"""
        agent = AgentFactory.create_agent("trend", "test_agent", 500000000)

        assert agent.get_max_position_size() == 0.15  # 15% position size
        assert agent.get_risk_tolerance() == "medium"

    def test_mean_reversion_strategy(self):
        """Test Mean Reversion agent properties"""
        agent = AgentFactory.create_agent("mean_reversion", "test_agent", 500000000)

        assert agent.get_max_position_size() == 0.12  # 12% position size
        assert agent.get_risk_tolerance() == "low"

    def test_strategy_position_limits(self):
        """Test that different strategies have appropriate position limits"""
        conservative = AgentFactory.create_agent(
            "conservative", "conservative", 500000000
        )
        aggressive = AgentFactory.create_agent("aggressive", "aggressive", 500000000)

        # Conservative should have smaller max position than aggressive
        assert conservative.get_max_position_size() < aggressive.get_max_position_size()

        # Test actual position execution respects limits
        max_conservative_value = 500000000 * conservative.get_max_position_size()
        max_aggressive_value = 500000000 * aggressive.get_max_position_size()

        assert max_conservative_value < max_aggressive_value
