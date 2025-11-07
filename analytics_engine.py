#!/usr/bin/env python3
"""
Analytics and Tuning Engine for AI Trading System
Uses collected trading data to improve AI performance
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class TradingAnalytics:
    def __init__(self, db_path="trading_analytics.db"):
        """Initialize analytics engine"""
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables for analytics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TEXT,
                agent_id INTEGER,
                strategy TEXT,
                starting_capital REAL,
                ending_capital REAL,
                total_trades INTEGER,
                winning_trades INTEGER,
                total_pnl REAL,
                win_rate REAL,
                sharpe_ratio REAL,
                max_drawdown REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS individual_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                agent_id INTEGER,
                timestamp TEXT,
                symbol TEXT,
                action TEXT,
                shares INTEGER,
                price REAL,
                value REAL,
                pnl REAL,
                market_conditions TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                change_percent REAL,
                technical_indicators TEXT
            )
        """)

        conn.commit()
        conn.close()
        logger.info("Analytics database initialized")

    def save_trading_session(self, session_data: Dict):
        """Save a complete trading session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO trading_sessions
            (session_date, agent_id, strategy, starting_capital, ending_capital,
             total_trades, winning_trades, total_pnl, win_rate, sharpe_ratio, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session_data["date"],
                session_data["agent_id"],
                session_data["strategy"],
                session_data["starting_capital"],
                session_data["ending_capital"],
                session_data["total_trades"],
                session_data["winning_trades"],
                session_data["total_pnl"],
                session_data["win_rate"],
                session_data.get("sharpe_ratio", 0),
                session_data.get("max_drawdown", 0),
            ),
        )

        session_id = cursor.lastrowid

        # Save individual trades
        for trade in session_data.get("trades", []):
            cursor.execute(
                """
                INSERT INTO individual_trades
                (session_id, agent_id, timestamp, symbol, action, shares, price, value, pnl, market_conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    trade["agent_id"],
                    trade["timestamp"],
                    trade["symbol"],
                    trade["action"],
                    trade["shares"],
                    trade["price"],
                    trade["value"],
                    trade.get("pnl", 0),
                    json.dumps(trade.get("market_conditions", {})),
                ),
            )

        conn.commit()
        conn.close()
        logger.info(
            f"Saved trading session {session_id} for agent {session_data['agent_id']}"
        )

    def analyze_strategy_performance(self, days_back: int = 30) -> Dict:
        """Analyze performance of different strategies"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT strategy,
                   AVG(win_rate) as avg_win_rate,
                   AVG(total_pnl) as avg_pnl,
                   AVG(sharpe_ratio) as avg_sharpe,
                   AVG(max_drawdown) as avg_drawdown,
                   COUNT(*) as session_count,
                   SUM(total_trades) as total_trades
            FROM trading_sessions
            WHERE session_date >= date('now', '-{} days')
            GROUP BY strategy
        """.format(days_back)

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return {}

        # Convert to dictionary
        performance = {}
        for _, row in df.iterrows():
            performance[row["strategy"]] = {
                "avg_win_rate": row["avg_win_rate"],
                "avg_pnl": row["avg_pnl"],
                "avg_sharpe": row["avg_sharpe"],
                "avg_drawdown": row["avg_drawdown"],
                "session_count": int(row["session_count"]),
                "total_trades": int(row["total_trades"]),
            }

        return performance

    def identify_best_performing_strategy(
        self, days_back: int = 30
    ) -> Tuple[str, Dict]:
        """Identify the best performing strategy"""
        performance = self.analyze_strategy_performance(days_back)

        if not performance:
            return "none", {}

        # Calculate composite score
        best_strategy = "none"
        best_score = -999

        for strategy, metrics in performance.items():
            # Composite score: 40% win_rate + 30% PnL + 20% Sharpe + 10% low drawdown
            score = (
                metrics["avg_win_rate"] * 0.4
                + (metrics["avg_pnl"] / 100) * 0.3  # Normalize PnL
                + metrics["avg_sharpe"] * 0.2
                + (1 - abs(metrics["avg_drawdown"]) / 100) * 0.1
            )

            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy, performance.get(best_strategy, {})

    def analyze_market_conditions_impact(self) -> Dict:
        """Analyze how different market conditions affect trading performance"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT
                t.action,
                t.pnl,
                json_extract(t.market_conditions, '$.volatility') as volatility,
                json_extract(t.market_conditions, '$.trend') as trend,
                json_extract(t.market_conditions, '$.volume') as volume,
                m.change_percent as market_change
            FROM individual_trades t
            JOIN market_snapshots m ON t.symbol = m.symbol
                                     AND DATE(t.timestamp) = DATE(m.timestamp)
            WHERE t.pnl IS NOT NULL
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return {}

        # Analyze by market conditions
        analysis = {}

        # By volatility
        df["volatility_bin"] = pd.cut(
            df["volatility"], bins=3, labels=["Low", "Medium", "High"]
        )
        volatility_analysis = df.groupby("volatility_bin")["pnl"].agg(["mean", "count"])
        analysis["by_volatility"] = volatility_analysis.to_dict()

        # By market trend
        df["trend_category"] = df["trend"].apply(
            lambda x: "Up" if x > 0.01 else "Down" if x < -0.01 else "Flat"
        )
        trend_analysis = df.groupby("trend_category")["pnl"].agg(["mean", "count"])
        analysis["by_trend"] = trend_analysis.to_dict()

        # By market change magnitude
        df["market_change_bin"] = pd.cut(
            df["market_change"],
            bins=5,
            labels=["Very Down", "Down", "Flat", "Up", "Very Up"],
        )
        change_analysis = df.groupby("market_change_bin")["pnl"].agg(["mean", "count"])
        analysis["by_market_change"] = change_analysis.to_dict()

        return analysis

    def generate_trading_signals(self, symbol: str, lookback_days: int = 30) -> Dict:
        """Generate trading signals based on historical performance"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT t.action, t.pnl, t.timestamp,
                   m.open_price, m.close_price, m.volume, m.change_percent
            FROM individual_trades t
            JOIN market_snapshots m ON t.symbol = m.symbol
                                     AND DATE(t.timestamp) = DATE(m.timestamp)
            WHERE t.symbol = ?
            AND t.timestamp >= date('now', '-{} days')
            ORDER BY t.timestamp
        """.format(lookback_days)

        df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()

        if df.empty or len(df) < 10:
            return {"signal": "NEUTRAL", "confidence": 0, "reason": "Insufficient data"}

        # Calculate signals based on historical performance
        buy_performance = df[df["action"] == "BUY"]["pnl"].mean()
        sell_performance = df[df["action"] == "SELL"]["pnl"].mean()

        # Recent performance (last 10 trades)
        recent_df = df.tail(10)
        recent_buy_perf = recent_df[recent_df["action"] == "BUY"]["pnl"].mean()
        recent_sell_perf = recent_df[recent_df["action"] == "SELL"]["pnl"].mean()

        # Market trend analysis
        avg_market_change = df["change_percent"].mean()
        volatility = df["change_percent"].std()

        # Generate signal
        signal = "NEUTRAL"
        confidence = 0
        reason = ""

        if recent_buy_perf > 0 and recent_sell_perf > 0:
            signal = "BUY"
            confidence = min(abs(recent_buy_perf) / 100, 1.0)
            reason = f"Recent buy performance: {recent_buy_perf:.2f}%"
        elif recent_buy_perf < 0 and recent_sell_perf < 0:
            signal = "SELL"
            confidence = min(abs(recent_sell_perf) / 100, 1.0)
            reason = f"Recent sell performance: {recent_sell_perf:.2f}%"
        elif avg_market_change > 2:
            signal = "BUY"
            confidence = min(avg_market_change / 10, 1.0)
            reason = f"Strong market trend: +{avg_market_change:.2f}%"
        elif avg_market_change < -2:
            signal = "SELL"
            confidence = min(abs(avg_market_change) / 10, 1.0)
            reason = f"Weak market trend: {avg_market_change:.2f}%"

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "historical_buy_performance": buy_performance,
            "historical_sell_performance": sell_performance,
            "avg_market_change": avg_market_change,
            "volatility": volatility,
            "data_points": len(df),
        }

    def create_ml_training_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create dataset for machine learning training"""
        conn = sqlite3.connect(self.db_path)

        # Get trading data with features
        query = """
            SELECT
                t.agent_id,
                t.strategy,
                t.action,
                t.pnl,
                t.timestamp,
                m.open_price,
                m.high_price,
                m.low_price,
                m.close_price,
                m.volume,
                m.change_percent,
                json_extract(t.market_conditions, '$.volatility') as volatility,
                json_extract(t.market_conditions, '$.trend') as trend
            FROM individual_trades t
            JOIN market_snapshots m ON t.symbol = m.symbol
                                     AND DATE(t.timestamp) = DATE(m.timestamp)
            WHERE t.pnl IS NOT NULL
            ORDER BY t.timestamp
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Create features
        df["return"] = (df["close_price"] - df["open_price"]) / df["open_price"]
        df["high_low_ratio"] = (df["high_price"] - df["low_price"]) / df["close_price"]
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        df["price_momentum"] = df["close_price"].pct_change(5)

        # Create target variable (successful trade vs unsuccessful)
        df["target"] = (df["pnl"] > 0).astype(int)

        # Feature selection
        feature_columns = [
            "return",
            "high_low_ratio",
            "volume_ratio",
            "price_momentum",
            "change_percent",
            "volatility",
            "trend",
        ]

        X = df[feature_columns].fillna(0)
        y = df["target"]

        return X, y

    def train_ml_model(self):
        """Train machine learning model for trade prediction"""
        X, y = self.create_ml_training_dataset()

        if X.empty:
            logger.warning("No data available for ML training")
            return None, 0

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

        model.fit(X_train, y_train)

        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))

        logger.info(
            f"ML Model trained - Train Score: {train_score:.3f}, Test Score: {test_score:.3f}"
        )

        return model, test_score

    def generate_performance_report(self, days_back: int = 30) -> str:
        """Generate comprehensive performance report"""
        best_strategy, best_metrics = self.identify_best_performing_strategy(days_back)
        strategy_performance = self.analyze_strategy_performance(days_back)
        market_analysis = self.analyze_market_conditions_impact()

        report = f"""
📊 TRADING PERFORMANCE REPORT (Last {days_back} Days)
{"=" * 60}

🏆 BEST PERFORMING STRATEGY: {best_strategy.upper()}
   Average Win Rate: {best_metrics.get("avg_win_rate", 0):.2%}
   Average P&L: {best_metrics.get("avg_pnl", 0):.2f}%
   Sharpe Ratio: {best_metrics.get("avg_sharpe", 0):.3f}
   Max Drawdown: {best_metrics.get("avg_drawdown", 0):.2f}%

📈 STRATEGY PERFORMANCE BREAKDOWN:
"""

        for strategy, metrics in strategy_performance.items():
            report += f"""
   {strategy}:
   - Win Rate: {metrics["avg_win_rate"]:.2%}
   - Avg P&L: {metrics["avg_pnl"]:.2f}%
   - Sessions: {metrics["session_count"]}
   - Total Trades: {metrics["total_trades"]}
"""

        if market_analysis:
            report += f"""
🌍 MARKET CONDITIONS ANALYSIS:
   Volatility Impact: {market_analysis.get("by_volatility", {})}
   Trend Impact: {market_analysis.get("by_trend", {})}
   Market Change Impact: {market_analysis.get("by_market_change", {})}
"""

        # Train ML model
        model, score = self.train_ml_model()
        if model:
            report += f"""
🤖 MACHINE LEARNING INSIGHTS:
   Model Accuracy: {score:.3f}
   Data Points Used: {len(self.create_ml_training_dataset()[0])}
   Recommendation: {"Use ML predictions" if score > 0.6 else "Need more data for reliable predictions"}
"""

        return report


# Usage example
if __name__ == "__main__":
    analytics = TradingAnalytics()

    # Generate sample session data
    sample_session = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "agent_id": 1,
        "strategy": "Conservative",
        "starting_capital": 500000000,
        "ending_capital": 512500000,
        "total_trades": 15,
        "winning_trades": 12,
        "total_pnl": 12500000,
        "win_rate": 0.8,
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.05,
        "trades": [
            {
                "agent_id": 1,
                "timestamp": datetime.now().isoformat(),
                "symbol": "BBCA.JK",
                "action": "BUY",
                "shares": 100,
                "price": 9800,
                "value": 980000,
                "pnl": 25000,
                "market_conditions": {
                    "volatility": 0.02,
                    "trend": 0.01,
                    "volume": 15000000,
                },
            }
        ],
    }

    analytics.save_trading_session(sample_session)

    # Generate performance report
    report = analytics.generate_performance_report()
    print(report)

    # Generate trading signals for specific stocks
    for symbol in ["BBCA.JK", "BBRI.JK", "TLKM.JK"]:
        signal = analytics.generate_trading_signals(symbol)
        print(
            f"\n📊 {symbol} Signal: {signal['signal']} (Confidence: {signal['confidence']:.2f})"
        )
        print(f"   Reason: {signal['reason']}")
