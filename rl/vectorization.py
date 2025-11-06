# rl/vectorization.py
"""Shared utilities for state and action vectorization in RL training"""

import numpy as np
from typing import Dict, Any


class StateVectorizer:
    """Consistent state vectorization for RL training"""

    def __init__(self, initial_capital: float = 500000000):
        self.initial_capital = initial_capital
        self.state_dim = 19  # Fixed dimension for consistency

    def state_dict_to_vector(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Convert state dictionary to normalized vector

        Args:
            state_dict: Dictionary containing state information

        Returns:
            np.ndarray: Normalized state vector of shape (20,)
        """
        state_vector = [
            # Financial metrics (normalized by initial capital)
            state_dict.get("cash_balance", 0) / self.initial_capital,
            state_dict.get("total_pnl", 0) / self.initial_capital,
            state_dict.get("available_margin", 0) / self.initial_capital,
            len(state_dict.get("positions", [])),
            # Technical indicators (normalized)
            state_dict.get("rsi", 50) / 100.0,
            state_dict.get("macd", 0) / 1.0,
            state_dict.get("bollinger_position", 0.5),
            state_dict.get("atr", 0) / 100.0,
            state_dict.get("volume_ratio", 1.0) / 10.0,
            state_dict.get("market_volatility", 0),
            # Time features (normalized)
            state_dict.get("hour_of_day", 12) / 24.0,
            state_dict.get("day_of_week", 3) / 7.0,
            # Market session one-hot encoding
            1.0 if state_dict.get("market_session") == "us" else 0.0,
            1.0 if state_dict.get("market_session") == "asia" else 0.0,
            1.0 if state_dict.get("market_session") == "europe" else 0.0,
            # Strategy type one-hot encoding (4 strategies)
            1.0 if state_dict.get("strategy_type") == "conservative" else 0.0,
            1.0 if state_dict.get("strategy_type") == "aggressive" else 0.0,
            1.0 if state_dict.get("strategy_type") == "balanced" else 0.0,
            1.0 if state_dict.get("strategy_type") == "trend" else 0.0,
        ]

        return np.array(state_vector, dtype=np.float32)


class ActionVectorizer:
    """Consistent action vectorization for RL training"""

    def __init__(self):
        self.action_dim = 2  # [action_type, position_size]

        # Action type mappings
        self.action_to_continuous = {"HOLD": 0.0, "BUY": 1.0, "SELL": 2.0}

        self.continuous_to_action = {0.0: "HOLD", 1.0: "BUY", 2.0: "SELL"}

    def action_dict_to_vector(self, action_dict: Dict[str, Any]) -> np.ndarray:
        """Convert action dictionary to vector

        Args:
            action_dict: Dictionary containing action information

        Returns:
            np.ndarray: Action vector of shape (2,)
        """
        action_type = action_dict.get("type", "HOLD")
        action_val = self.action_to_continuous.get(action_type, 0.0)

        # Position size (0.0 to 1.0)
        position_size = action_dict.get("position_size", 0.5)
        position_size = np.clip(position_size, 0.0, 1.0)

        return np.array([action_val, position_size], dtype=np.float32)

    def vector_to_action_dict(self, action_vector: np.ndarray) -> Dict[str, Any]:
        """Convert vector back to action dictionary

        Args:
            action_vector: Action vector of shape (2,)

        Returns:
            Dict[str, Any]: Action dictionary
        """
        action_type_val = float(action_vector[0])
        position_size = float(action_vector[1])

        # Determine action type
        if action_type_val < 0.5:
            action_type = "HOLD"
        elif action_type_val < 1.5:
            action_type = "BUY"
        else:
            action_type = "SELL"

        # Clamp position size
        position_size = np.clip(position_size, 0.0, 1.0)

        return {"type": action_type, "position_size": position_size}


# Global instances for easy import
state_vectorizer = StateVectorizer()
action_vectorizer = ActionVectorizer()
