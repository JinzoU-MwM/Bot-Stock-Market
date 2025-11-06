import uuid
from datetime import datetime
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, name, strategy_type, initial_capital):
        self.id = str(uuid.uuid4())
        self.name = name
        self.strategy_type = strategy_type
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self.positions = {}  # symbol: {'quantity': int, 'entry_price': float, 'entry_time': datetime}
        self.trades = []
        self.status = "active"
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0

    @abstractmethod
    def get_max_position_size(self):
        """Get maximum position size as percentage of capital"""
        pass

    @abstractmethod
    def get_risk_tolerance(self):
        """Get risk tolerance level"""
        pass

    def get_state(self, symbol, market_data=None):
        """Get current state representation for RL"""
        # Portfolio metrics
        total_position_value = sum(
            pos["quantity"] * pos["entry_price"] for pos in self.positions.values()
        )
        total_pnl = self.calculate_total_pnl()

        state = {
            # Portfolio metrics
            "cash_balance": self.current_balance,
            "positions": list(self.positions.values()),
            "total_pnl": total_pnl,
            "available_margin": self.current_balance,
            "portfolio_value": self.current_balance + total_position_value + total_pnl,
            # Technical indicators (from market_data)
            "rsi": market_data.get("rsi", 50.0) if market_data else 50.0,
            "macd": market_data.get("macd", 0.0) if market_data else 0.0,
            "bollinger_position": 0.0,  # Will be calculated
            "atr": market_data.get("atr", 0.0) if market_data else 0.0,
            "volume_ratio": market_data.get("volume_ratio", 1.0)
            if market_data
            else 1.0,
            # Market sentiment
            "market_volatility": market_data.get("atr", 0.0)
            / market_data.get("price", 1.0)
            if market_data
            else 0.0,
            "sentiment_score": 0.0,  # Placeholder for future sentiment analysis
            # Time features
            "hour_of_day": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "market_session": self.get_market_session(),
        }

        # Calculate Bollinger Band position
        if market_data and all(
            k in market_data for k in ["bollinger_upper", "bollinger_lower", "price"]
        ):
            bb_range = market_data["bollinger_upper"] - market_data["bollinger_lower"]
            if bb_range > 0:
                state["bollinger_position"] = (
                    market_data["price"] - market_data["bollinger_lower"]
                ) / bb_range

        return state

    def get_market_session(self):
        """Determine current market session"""
        hour = datetime.now().hour

        if 9 <= hour <= 16:
            return "us"  # US market hours
        elif hour <= 8:
            return "asia"  # Asian market hours
        else:
            return "europe"  # European market hours

    def calculate_total_pnl(self):
        """Calculate total P&L including unrealized gains"""
        total_pnl = 0.0

        # Realized P&L from closed trades
        for trade in self.trades:
            if trade.get("profit_loss"):
                total_pnl += trade["profit_loss"]

        # TODO: Add unrealized P&L from open positions
        # This would require current market prices for open positions

        return total_pnl

    def execute_action(self, action, market_data=None):
        """Execute trading action"""
        try:
            action_type = action["type"]
            symbol = action["symbol"]
            quantity = action.get("quantity", 0)
            price = action.get("price", market_data.get("price") if market_data else 0)

            if action_type == "BUY":
                return self._execute_buy(symbol, quantity, price, market_data)
            elif action_type == "SELL":
                return self._execute_sell(symbol, quantity, price, market_data)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action_type}",
                }

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {"success": False, "error": str(e)}

    def _execute_buy(self, symbol, quantity, price, market_data):
        """Execute buy order"""
        required_capital = quantity * price
        commission = required_capital * 0.001  # 0.1% commission
        total_cost = required_capital + commission

        if self.current_balance < total_cost:
            return {"success": False, "error": "Insufficient funds"}

        # Check position limits
        if symbol in self.positions:
            return {"success": False, "error": "Position already exists"}

        if len(self.positions) >= 3:  # Max 3 positions per agent
            return {"success": False, "error": "Maximum positions reached"}

        # Execute trade
        self.current_balance -= total_cost

        self.positions[symbol] = {
            "quantity": quantity,
            "entry_price": price,
            "entry_time": datetime.now(),
        }

        trade_record = {
            "id": str(uuid.uuid4()),
            "type": "BUY",
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "commission": commission,
            "timestamp": datetime.now(),
            "step": self.step_count,
            "episode_id": self.episode_id,
        }

        self.trades.append(trade_record)

        logger.info(
            f"Agent {self.name} bought {quantity} shares of {symbol} at ${price}"
        )

        return {
            "success": True,
            "trade": trade_record,
            "new_balance": self.current_balance,
            "positions_count": len(self.positions),
        }

    def _execute_sell(self, symbol, quantity, price, market_data):
        """Execute sell order"""
        if symbol not in self.positions:
            return {"success": False, "error": "No position to sell"}

        position = self.positions[symbol]
        if position["quantity"] < quantity:
            return {"success": False, "error": "Insufficient position quantity"}

        # Calculate P&L
        sale_proceeds = quantity * price
        commission = sale_proceeds * 0.001
        net_proceeds = sale_proceeds - commission
        cost_basis = quantity * position["entry_price"]
        profit_loss = net_proceeds - cost_basis

        # Execute trade
        self.current_balance += net_proceeds

        # Update or remove position
        if position["quantity"] == quantity:
            del self.positions[symbol]
        else:
            position["quantity"] -= quantity

        trade_record = {
            "id": str(uuid.uuid4()),
            "type": "SELL",
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "commission": commission,
            "profit_loss": profit_loss,
            "timestamp": datetime.now(),
            "step": self.step_count,
            "episode_id": self.episode_id,
        }

        self.trades.append(trade_record)

        logger.info(
            f"Agent {self.name} sold {quantity} shares of {symbol} at ${price}, P&L: ${profit_loss:.2f}"
        )

        return {
            "success": True,
            "trade": trade_record,
            "profit_loss": profit_loss,
            "new_balance": self.current_balance,
            "positions_count": len(self.positions),
        }

    def get_performance_metrics(self):
        """Get current performance metrics"""
        total_trades = len([t for t in self.trades if t["type"] == "SELL"])
        winning_trades = len([t for t in self.trades if t.get("profit_loss", 0) > 0])
        losing_trades = len([t for t in self.trades if t.get("profit_loss", 0) < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = self.calculate_total_pnl()

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "current_balance": self.current_balance,
            "positions_count": len(self.positions),
            "total_return": (self.current_balance - self.initial_capital)
            / self.initial_capital,
        }
