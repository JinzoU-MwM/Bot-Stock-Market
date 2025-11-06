# rl/environment.py
import gym
from gym import spaces
import numpy as np
from typing import Dict, Any, Tuple
import uuid
import logging

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    def __init__(self, agent, market_data_service, database):
        super().__init__()

        self.agent = agent
        self.market_data_service = market_data_service
        self.database = database
        self.current_step = 0
        self.max_steps = 1000

        # Action space: [action_type, position_size]
        # action_type: 0=HOLD, 1=BUY, 2=SELL
        # position_size: 0.0 to 1.0 (fraction of max position size)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]), high=np.array([2.0, 1.0]), dtype=np.float32
        )

        # State space (simplified for now)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),  # Will expand as needed
            dtype=np.float32,
        )

        self.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]  # Default symbols
        self.current_symbol_index = 0
        self.current_symbol = self.symbols[0]

    def reset(self):
        """Reset environment for new episode"""
        self.current_step = 0
        self.current_symbol_index = 0
        self.current_symbol = self.symbols[0]

        # Reset agent state
        self.agent.episode_id = str(uuid.uuid4())
        self.agent.step_count = 0

        # Get initial state
        state = self._get_current_state()
        return state

    def step(self, action):
        """Execute one step in environment"""
        self.current_step += 1
        self.agent.step_count += 1

        # Execute action
        action_type, position_size = action
        reward, done, info = self._execute_action(action_type, position_size)

        # Get next state
        next_state = self._get_current_state()

        # Check if episode should end
        if self.current_step >= self.max_steps:
            done = True

        return next_state, reward, done, info

    def _get_current_state(self):
        """Get current state representation"""
        market_data = self.market_data_service.calculate_technical_indicators(
            self.current_symbol
        )
        state_dict = self.agent.get_state(self.current_symbol, market_data)

        # Convert dict to numpy array for RL model
        state_vector = [
            state_dict["cash_balance"] / self.agent.initial_capital,
            state_dict["total_pnl"] / self.agent.initial_capital,
            state_dict["available_margin"] / self.agent.initial_capital,
            len(state_dict["positions"]),
            state_dict["rsi"] / 100.0 if state_dict["rsi"] else 0.5,
            state_dict["macd"] / 1.0 if state_dict["macd"] else 0.0,
            state_dict["bollinger_position"]
            if state_dict["bollinger_position"]
            else 0.5,
            state_dict["atr"] / 100.0 if state_dict["atr"] else 0.0,
            state_dict["volume_ratio"] / 10.0 if state_dict["volume_ratio"] else 1.0,
            state_dict["market_volatility"],
            state_dict["hour_of_day"] / 24.0,
            state_dict["day_of_week"] / 7.0,
            # One-hot encoding for market session
            1.0 if state_dict["market_session"] == "us" else 0.0,
            1.0 if state_dict["market_session"] == "asia" else 0.0,
            1.0 if state_dict["market_session"] == "europe" else 0.0,
            # Strategy type encoding
            1.0 if self.agent.strategy_type == "conservative" else 0.0,
            1.0 if self.agent.strategy_type == "aggressive" else 0.0,
            1.0 if self.agent.strategy_type == "balanced" else 0.0,
            1.0 if self.agent.strategy_type == "trend" else 0.0,
            1.0 if self.agent.strategy_type == "mean_reversion" else 0.0,
        ]

        return np.array(state_vector, dtype=np.float32)

    def _execute_action(self, action_type, position_size):
        """Execute trading action and calculate reward"""
        market_data = self.market_data_service.get_real_time_data(self.current_symbol)

        if not market_data:
            return -0.1, False, {"error": "No market data available"}

        # Determine action
        if action_type < 0.5:  # HOLD
            return self._calculate_hold_reward(), False, {"action": "HOLD"}

        elif action_type < 1.5:  # BUY
            return self._execute_buy_action(position_size, market_data)

        else:  # SELL
            return self._execute_sell_action(position_size, market_data)

    def _execute_buy_action(self, position_size, market_data):
        """Execute buy action"""
        max_position_value = (
            self.agent.initial_capital * self.agent.get_max_position_size()
        )
        position_value = max_position_value * position_size
        price = market_data["price"]
        quantity = int(position_value / price)

        if quantity <= 0:
            return -0.01, False, {"action": "BUY", "result": "invalid_quantity"}

        action = {
            "type": "BUY",
            "symbol": self.current_symbol,
            "quantity": quantity,
            "price": price,
        }

        result = self.agent.execute_action(action, market_data)

        if result["success"]:
            reward = 0.01  # Small positive reward for successful execution
            # Store to database
            self._store_experience_to_db(result["trade"])
            return (
                reward,
                False,
                {"action": "BUY", "result": "success", "trade": result["trade"]},
            )
        else:
            return (
                -0.05,
                False,
                {"action": "BUY", "result": "failed", "error": result.get("error")},
            )

    def _execute_sell_action(self, position_size, market_data):
        """Execute sell action"""
        if self.current_symbol not in self.agent.positions:
            return -0.02, False, {"action": "SELL", "result": "no_position"}

        position = self.agent.positions[self.current_symbol]
        max_sellable = position["quantity"]
        sell_quantity = int(max_sellable * position_size)

        if sell_quantity <= 0:
            return -0.01, False, {"action": "SELL", "result": "invalid_quantity"}

        action = {
            "type": "SELL",
            "symbol": self.current_symbol,
            "quantity": sell_quantity,
            "price": market_data["price"],
        }

        result = self.agent.execute_action(action, market_data)

        if result["success"]:
            profit_loss = result.get("profit_loss", 0)
            # Calculate reward based on profit/loss
            reward = profit_loss / self.agent.initial_capital  # Normalize by capital
            # Store to database
            self._store_experience_to_db(result["trade"])
            return (
                reward,
                False,
                {"action": "SELL", "result": "success", "profit_loss": profit_loss},
            )
        else:
            return (
                -0.05,
                False,
                {"action": "SELL", "result": "failed", "error": result.get("error")},
            )

    def _calculate_hold_reward(self):
        """Calculate reward for holding"""
        # Small negative reward for inactivity (opportunity cost)
        return -0.001

    def _store_experience_to_db(self, trade):
        """Store trade/experience to database"""
        try:
            # Store to trades table
            query = """
                INSERT INTO trades
                (agent_id, symbol, action, quantity, price, timestamp, profit_loss, commission, strategy_reasoning)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            params = (
                self.agent.id,
                trade["symbol"],
                trade["type"],
                trade["quantity"],
                trade["price"],
                trade["timestamp"],
                trade.get("profit_loss", 0),
                trade.get("commission", 0),
                f"RL Agent - {self.agent.strategy_type} strategy",
            )

            self.database.execute(query, params)

        except Exception as e:
            logger.error(f"Failed to store trade to database: {e}")
