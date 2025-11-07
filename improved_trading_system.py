#!/usr/bin/env python3
"""
IMPROVED Indonesian Stock Market Trading System
Fixed AI agent trading and dashboard visibility issues
"""

from flask import Flask, render_template_string, jsonify, request
import yfinance as yf
import pandas as pd
import json
import time
import logging
import random
from datetime import datetime, timedelta
import traceback
import math
import sqlite3
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Indonesian stocks with realistic base prices
INDONESIAN_STOCKS = [
    # Banking Sector
    {
        "symbol": "BBCA.JK",
        "name": "Bank Central Asia",
        "sector": "Banking",
        "base_price": 9750,
    },
    {"symbol": "BBRI.JK", "name": "Bank BRI", "sector": "Banking", "base_price": 4890},
    {
        "symbol": "BMRI.JK",
        "name": "Bank Mandiri",
        "sector": "Banking",
        "base_price": 6850,
    },
    {"symbol": "BBNI.JK", "name": "Bank BNI", "sector": "Banking", "base_price": 5250},
    # Telecommunication
    {
        "symbol": "TLKM.JK",
        "name": "Telkom Indonesia",
        "sector": "Telecom",
        "base_price": 3450,
    },
    {"symbol": "EXCL.JK", "name": "XL Axiata", "sector": "Telecom", "base_price": 2650},
    {"symbol": "ISAT.JK", "name": "Indosat", "sector": "Telecom", "base_price": 6750},
    {"symbol": "FREN.JK", "name": "Smartfren", "sector": "Telecom", "base_price": 78},
    # Consumer Goods
    {
        "symbol": "UNVR.JK",
        "name": "Unilever Indonesia",
        "sector": "Consumer",
        "base_price": 4250,
    },
    {"symbol": "INDF.JK", "name": "Indofood", "sector": "Consumer", "base_price": 6250},
    {
        "symbol": "ICBP.JK",
        "name": "Indofood CBP",
        "sector": "Consumer",
        "base_price": 9850,
    },
    {
        "symbol": "KLBF.JK",
        "name": "Kalbe Farma",
        "sector": "Consumer",
        "base_price": 1750,
    },
    {
        "symbol": "MYOR.JK",
        "name": "Mayora Indah",
        "sector": "Consumer",
        "base_price": 2250,
    },
    # Automotive
    {
        "symbol": "ASII.JK",
        "name": "Astra International",
        "sector": "Automotive",
        "base_price": 6750,
    },
    # Construction
    {
        "symbol": "ADHI.JK",
        "name": "Adhi Karya",
        "sector": "Construction",
        "base_price": 850,
    },
    {
        "symbol": "WIKA.JK",
        "name": "Wijaya Karya",
        "sector": "Construction",
        "base_price": 650,
    },
    {
        "symbol": "PTPP.JK",
        "name": "Waskita Karya",
        "sector": "Construction",
        "base_price": 450,
    },
    # Mining
    {
        "symbol": "ANTM.JK",
        "name": "Aneka Tambang",
        "sector": "Mining",
        "base_price": 2850,
    },
    {
        "symbol": "SMIN.JK",
        "name": "Sri Rejeki Isman",
        "sector": "Mining",
        "base_price": 350,
    },
    {
        "symbol": "HRUM.JK",
        "name": "Harum Energy",
        "sector": "Mining",
        "base_price": 1250,
    },
    {
        "symbol": "ADRO.JK",
        "name": "Adaro Energy",
        "sector": "Mining",
        "base_price": 1450,
    },
    {
        "symbol": "ITMG.JK",
        "name": "Indo Tambangraya Megah",
        "sector": "Mining",
        "base_price": 22500,
    },
    {"symbol": "PTBA.JK", "name": "Bukit Asam", "sector": "Mining", "base_price": 3350},
    {"symbol": "TINS.JK", "name": "Timah", "sector": "Mining", "base_price": 1850},
    # Property
    {
        "symbol": "BSDE.JK",
        "name": "Bumi Serpong Damai",
        "sector": "Property",
        "base_price": 1350,
    },
    {
        "symbol": "ASRI.JK",
        "name": "Alam Sutera Realty",
        "sector": "Property",
        "base_price": 350,
    },
    {
        "symbol": "PWON.JK",
        "name": "Pakuwon Jati",
        "sector": "Property",
        "base_price": 450,
    },
    {
        "symbol": "LPKR.JK",
        "name": "Lippo Karawaci",
        "sector": "Property",
        "base_price": 250,
    },
    {
        "symbol": "CTRA.JK",
        "name": "Ciputra Development",
        "sector": "Property",
        "base_price": 1150,
    },
    # Infrastructure
    {
        "symbol": "JSMR.JK",
        "name": "Jasa Marga",
        "sector": "Infrastructure",
        "base_price": 4750,
    },
    {
        "symbol": "PGAS.JK",
        "name": "Perusahaan Gas Negara",
        "sector": "Infrastructure",
        "base_price": 1850,
    },
    {
        "symbol": "TOWR.JK",
        "name": "Sarana Menara Nusantara",
        "sector": "Infrastructure",
        "base_price": 1250,
    },
    # Technology
    {
        "symbol": "TCID.JK",
        "name": "MNC Investama",
        "sector": "Technology",
        "base_price": 95,
    },
    {
        "symbol": "MNCN.JK",
        "name": "Media Nusantara Citra",
        "sector": "Technology",
        "base_price": 750,
    },
    {
        "symbol": "BKLA.JK",
        "name": "Bukalapak",
        "sector": "Technology",
        "base_price": 450,
    },
    {
        "symbol": "BIRD.JK",
        "name": "Blue Bird",
        "sector": "Technology",
        "base_price": 2250,
    },
    {
        "symbol": "GIAA.JK",
        "name": "Garuda Indonesia",
        "sector": "Technology",
        "base_price": 85,
    },
    # Tobacco
    {
        "symbol": "GGRM.JK",
        "name": "Gudang Garam",
        "sector": "Tobacco",
        "base_price": 37500,
    },
    {
        "symbol": "HMSP.JK",
        "name": "HM Sampoerna",
        "sector": "Tobacco",
        "base_price": 1250,
    },
    # Other
    {
        "symbol": "BYAN.JK",
        "name": "Bayan Resources",
        "sector": "Mining",
        "base_price": 19500,
    },
    {
        "symbol": "DOID.JK",
        "name": "Delta Dunia Makmur",
        "sector": "Mining",
        "base_price": 150,
    },
]

# Agent data structure with different strategies
agents = {
    1: {
        "cash": 500000000,
        "positions": {},
        "trades": [],
        "total_trades": 0,
        "winning_trades": 0,
        "status": "idle",
        "last_action": "Menunggu data market...",
        "strategy": "Conservative",
        "color": "#4CAF50",
    },
    2: {
        "cash": 500000000,
        "positions": {},
        "trades": [],
        "total_trades": 0,
        "winning_trades": 0,
        "status": "idle",
        "last_action": "Menunggu data market...",
        "strategy": "Aggressive",
        "color": "#f44336",
    },
    3: {
        "cash": 500000000,
        "positions": {},
        "trades": [],
        "total_trades": 0,
        "winning_trades": 0,
        "status": "idle",
        "last_action": "Menunggu data market...",
        "strategy": "Balanced",
        "color": "#2196f3",
    },
    4: {
        "cash": 500000000,
        "positions": {},
        "trades": [],
        "total_trades": 0,
        "winning_trades": 0,
        "status": "idle",
        "last_action": "Menunggu data market...",
        "strategy": "Trend",
        "color": "#ff9800",
    },
    5: {
        "cash": 500000000,
        "positions": {},
        "trades": [],
        "total_trades": 0,
        "winning_trades": 0,
        "status": "idle",
        "last_action": "Menunggu data market...",
        "strategy": "Mean Reversion",
        "color": "#9c27b0",
    },
}

agents_active = False
market_cache = {}
cache_time = 0
api_working = False
simulation_mode = True

# Analytics Engine
trading_sessions_db = "trading_sessions.db"
current_session_data = {}
session_start_time = None


def init_analytics_database():
    """Initialize analytics database for trading data"""
    conn = sqlite3.connect(trading_sessions_db)
    cursor = conn.cursor()

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
            max_drawdown REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS individual_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            agent_id INTEGER,
            strategy TEXT,
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

    conn.commit()
    conn.close()
    logger.info("📊 Analytics database initialized")


def start_trading_session():
    """Start a new trading session"""
    global current_session_data, session_start_time

    session_start_time = datetime.now()
    current_session_data = {
        "session_date": session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "agents": {},
    }

    # Initialize data for each agent
    for agent_id in agents:
        current_session_data["agents"][agent_id] = {
            "agent_id": agent_id,
            "strategy": agents[agent_id]["strategy"],
            "starting_capital": 500000000,
            "trades": [],
            "starting_time": session_start_time,
            "capital_history": [(session_start_time, 500000000)],
        }

    logger.info(f"📊 Trading session started at {session_start_time}")


def end_trading_session():
    """End current trading session and save to database"""
    global current_session_data, session_start_time

    if not current_session_data or not session_start_time:
        return

    end_time = datetime.now()
    session_duration = (end_time - session_start_time).total_seconds() / 60  # minutes

    # Calculate final metrics for each agent
    for agent_id, agent_data in current_session_data["agents"].items():
        agent = agents[agent_id]
        portfolio = calculate_portfolio_value(agent_id, market_cache)

        # Calculate additional metrics
        total_pnl = portfolio["pnl"]
        win_rate = agent["winning_trades"] / max(agent["total_trades"], 1)

        # Calculate Sharpe ratio (simplified)
        returns = [trade.get("pnl", 0) / 500000000 for trade in agent_data["trades"]]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) if returns else 0

        # Calculate max drawdown
        max_drawdown = 0
        if len(agent_data["capital_history"]) > 1:
            peak = max(h[1] for h in agent_data["capital_history"])
            current = portfolio["total_value"]
            max_drawdown = (current - peak) / peak if peak > 0 else 0

        # Save session to database
        conn = sqlite3.connect(trading_sessions_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO trading_sessions
            (session_date, agent_id, strategy, starting_capital, ending_capital,
             total_trades, winning_trades, total_pnl, win_rate, sharpe_ratio, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                agent_data["session_date"],
                agent_id,
                agent_data["strategy"],
                agent_data["starting_capital"],
                portfolio["total_value"],
                agent["total_trades"],
                agent["winning_trades"],
                total_pnl,
                win_rate,
                sharpe_ratio,
                max_drawdown,
            ),
        )

        session_id = cursor.lastrowid

        # Save individual trades
        for trade in agent_data["trades"]:
            cursor.execute(
                """
                INSERT INTO individual_trades
                (session_id, agent_id, timestamp, symbol, action, shares, price, value, pnl, market_conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    agent_id,
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

        logger.info(f"💾 Session saved for Agent {agent_id}: {agent_data['strategy']}")
        logger.info(f"   Total P&L: Rp {total_pnl:,.2f}, Win Rate: {win_rate:.2%}")

    # Generate performance report
    generate_daily_report()

    current_session_data = {}
    session_start_time = None
    logger.info(f"📊 Trading session ended after {session_duration:.1f} minutes")


def record_trade(agent_id, trade_data):
    """Record a trade in current session"""
    global current_session_data

    if current_session_data and agent_id in current_session_data["agents"]:
        current_session_data["agents"][agent_id]["trades"].append(trade_data)

        # Update capital history
        portfolio = calculate_portfolio_value(agent_id, market_cache)
        current_session_data["agents"][agent_id]["capital_history"].append(
            (datetime.now(), portfolio["total_value"])
        )


def generate_daily_report():
    """Generate daily performance report"""
    try:
        conn = sqlite3.connect(trading_sessions_db)

        # Get today's sessions
        today = datetime.now().strftime("%Y-%m-%d")
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT strategy, AVG(win_rate) as avg_win_rate, AVG(total_pnl) as avg_pnl,
                   COUNT(*) as sessions, SUM(total_trades) as total_trades
            FROM trading_sessions
            WHERE session_date LIKE ?
            GROUP BY strategy
        """,
            (f"{today}%",),
        )

        results = cursor.fetchall()
        conn.close()

        if results:
            logger.info("📈 DAILY PERFORMANCE REPORT:")
            logger.info("=" * 50)
            for row in results:
                logger.info(f"🤖 {row[0]} Strategy:")
                logger.info(f"   Avg Win Rate: {row[1]:.2%}")
                logger.info(f"   Avg P&L: Rp {row[2]:,.2f}")
                logger.info(f"   Sessions: {row[3]}")
                logger.info(f"   Total Trades: {row[4]}")
                logger.info("")

    except Exception as e:
        logger.error(f"Error generating daily report: {e}")


def get_best_performing_strategy(days_back=7):
    """Get the best performing strategy from analytics data"""
    try:
        conn = sqlite3.connect(trading_sessions_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT strategy, AVG(win_rate) as avg_win_rate, AVG(total_pnl) as avg_pnl
            FROM trading_sessions
            WHERE session_date >= date('now', '-{} days')
            GROUP BY strategy
            ORDER BY (AVG(win_rate) * 0.6 + AVG(total_pnl)/100000 * 0.4) DESC
            LIMIT 1
        """.format(days_back)
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                "strategy": result[0],
                "avg_win_rate": result[1],
                "avg_pnl": result[2],
            }
        return None

    except Exception as e:
        logger.error(f"Error getting best strategy: {e}")
        return None


def get_market_signals():
    """Generate trading signals based on historical performance"""
    try:
        conn = sqlite3.connect(trading_sessions_db)

        # Get recent successful trades
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.symbol, t.action, COUNT(*) as count, AVG(t.pnl) as avg_pnl
            FROM individual_trades t
            WHERE t.pnl > 0 AND t.created_at >= datetime('now', '-7 days')
            GROUP BY t.symbol, t.action
            HAVING count >= 3
            ORDER BY avg_pnl DESC
        """)

        results = cursor.fetchall()
        conn.close()

        signals = {}
        for symbol, action, count, avg_pnl in results:
            if symbol not in signals:
                signals[symbol] = {}
            signals[symbol][action] = {
                "confidence": min(count / 10, 1.0),
                "avg_pnl": avg_pnl,
                "sample_size": count,
            }

        return signals

    except Exception as e:
        logger.error(f"Error generating market signals: {e}")
        return {}


# Initialize analytics database
init_analytics_database()

# Simulated price history for realistic movements
price_history = {}
for stock in INDONESIAN_STOCKS:
    price_history[stock["symbol"]] = {
        "prices": [stock["base_price"]],
        "trend": random.choice([-0.01, 0, 0.01]),
        "volatility": random.uniform(0.02, 0.08),
    }


def try_yahoo_finance_connection():
    """Try to connect to Yahoo Finance API"""
    try:
        logger.info("Testing Yahoo Finance API connection...")
        ticker = yf.Ticker("BBCA.JK")
        hist = ticker.history(period="1d")

        if not hist.empty:
            logger.info("✅ Yahoo Finance API connection successful")
            return True
        else:
            logger.warning("⚠️ Yahoo Finance API returned empty data")
            return False

    except Exception as e:
        logger.error(f"❌ Yahoo Finance API connection failed: {e}")
        return False


def get_real_market_data():
    """Get market data from Yahoo Finance with fallback to realistic simulation"""
    global market_cache, cache_time, api_working, simulation_mode

    try:
        current_time = time.time()
        if market_cache and (current_time - cache_time) < 30:
            logger.info(f"Using cached market data from {len(market_cache)} stocks")
            return {
                "success": True,
                "data": market_cache,
                "cached": True,
                "source": "cache",
                "simulation_mode": simulation_mode,
            }

        if not simulation_mode:
            logger.info("Attempting Yahoo Finance API...")
            api_result = fetch_yahoo_finance_data()

            if api_result["success"] and len(api_result["data"]) > 10:
                api_working = True
                market_cache = api_result["data"]
                cache_time = current_time
                logger.info(
                    f"✅ Using REAL Yahoo Finance data: {len(api_result['data'])} stocks"
                )
                return {
                    "success": True,
                    "data": market_cache,
                    "cached": False,
                    "source": "Yahoo Finance API",
                    "simulation_mode": False,
                    "total_stocks": len(market_cache),
                }
            else:
                logger.warning(
                    "Yahoo Finance API failed, switching to realistic simulation"
                )
                simulation_mode = True
                api_working = False

        logger.info("Generating realistic market simulation...")
        market_data = generate_realistic_simulation()

        market_cache = market_data
        cache_time = current_time

        return {
            "success": True,
            "data": market_data,
            "cached": False,
            "source": "Realistic Market Simulation",
            "simulation_mode": True,
            "total_stocks": len(market_data),
        }

    except Exception as e:
        logger.error(f"❌ Error getting market data: {e}")

        if market_cache:
            logger.info("🔄 Returning cached data due to error")
            return {
                "success": True,
                "data": market_cache,
                "cached": True,
                "error": str(e),
                "source": "cache (error fallback)",
                "simulation_mode": simulation_mode,
            }

        emergency_data = generate_realistic_simulation()
        return {
            "success": True,
            "data": emergency_data,
            "cached": False,
            "source": "Emergency Simulation",
            "simulation_mode": True,
            "error": str(e),
        }


def fetch_yahoo_finance_data():
    """Fetch real data from Yahoo Finance API"""
    try:
        market_data = {}
        successful_fetches = 0

        for stock in INDONESIAN_STOCKS:
            symbol = stock["symbol"]
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d", interval="1d")

                if len(hist) >= 1:
                    latest = hist.iloc[-1]

                    if len(hist) >= 2:
                        previous = hist.iloc[-2]
                        prev_close = previous["Close"]
                    else:
                        prev_close = latest["Open"]

                    current_price = float(latest["Close"])
                    change = current_price - prev_close
                    change_percent = (
                        (change / prev_close) * 100 if prev_close > 0 else 0
                    )

                    market_data[symbol] = {
                        "symbol": symbol,
                        "price": current_price,
                        "open": float(latest["Open"]),
                        "high": float(latest["High"]),
                        "low": float(latest["Low"]),
                        "close": current_price,
                        "volume": max(1000000, int(latest["Volume"])),
                        "change": change,
                        "change_percent": change_percent,
                        "name": stock["name"],
                        "sector": stock["sector"],
                        "prev_close": prev_close,
                        "timestamp": time.time(),
                        "currency": "IDR",
                    }

                    successful_fetches += 1

            except Exception as e:
                logger.error(f"❌ Error fetching {symbol}: {e}")
                continue

        if len(market_data) > 0:
            logger.info(f"📊 Fetched real data for {len(market_data)} stocks")
            return {
                "success": True,
                "data": market_data,
                "successful_fetches": successful_fetches,
            }
        else:
            return {
                "success": False,
                "error": "No real data available from Yahoo Finance",
            }

    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_realistic_simulation():
    """Generate realistic market simulation based on Indonesian stock patterns"""
    market_data = {}

    market_sentiment = random.gauss(0, 0.015)
    sector_effects = {
        "Banking": random.gauss(0, 0.02),
        "Telecom": random.gauss(0, 0.018),
        "Consumer": random.gauss(0, 0.015),
        "Mining": random.gauss(0, 0.035),
        "Property": random.gauss(0, 0.025),
        "Technology": random.gauss(0, 0.04),
        "Automotive": random.gauss(0, 0.022),
        "Construction": random.gauss(0, 0.02),
        "Infrastructure": random.gauss(0, 0.016),
        "Tobacco": random.gauss(0, 0.01),
    }

    for stock in INDONESIAN_STOCKS:
        symbol = stock["symbol"]
        history = price_history[symbol]

        base_change = random.gauss(0, history["volatility"])
        sector_effect = sector_effects.get(stock["sector"], 0)
        trend_effect = history["trend"]

        total_change = base_change + sector_effect + market_sentiment + trend_effect
        total_change = max(-0.10, min(0.10, total_change))

        last_price = history["prices"][-1]
        new_price = last_price * (1 + total_change)
        new_price = max(new_price, stock["base_price"] * 0.3)

        high = new_price * (1 + abs(random.gauss(0, 0.01)))
        low = new_price * (1 - abs(random.gauss(0, 0.01)))
        open_price = last_price * (1 + random.gauss(0, 0.005))

        high = max(high, new_price, open_price)
        low = min(low, new_price, open_price)

        volume_base = random.uniform(500000, 5000000)
        if abs(total_change) > 0.03:
            volume_base *= random.uniform(1.5, 3)

        history["prices"].append(new_price)
        if len(history["prices"]) > 100:
            history["prices"] = history["prices"][-100:]

        if len(history["prices"]) >= 10:
            recent_avg = sum(history["prices"][-10:]) / 10
            base_price = stock["base_price"]
            if recent_avg > base_price * 1.1:
                history["trend"] = max(-0.02, history["trend"] - 0.001)
            elif recent_avg < base_price * 0.9:
                history["trend"] = min(0.02, history["trend"] + 0.001)

        market_data[symbol] = {
            "symbol": symbol,
            "price": new_price,
            "open": open_price,
            "high": high,
            "low": low,
            "close": new_price,
            "volume": int(volume_base),
            "change": new_price - last_price,
            "change_percent": total_change * 100,
            "name": stock["name"],
            "sector": stock["sector"],
            "prev_close": last_price,
            "timestamp": time.time(),
            "currency": "IDR",
            "base_price": stock["base_price"],
        }

    return market_data


def execute_trade(agent_id, symbol, action, shares, price):
    """Execute trade using current market prices"""
    try:
        agent = agents[agent_id]
        trade_value = shares * price

        if action == "BUY":
            if agent["cash"] >= trade_value:
                agent["cash"] -= trade_value

                if symbol in agent["positions"]:
                    old_shares = agent["positions"][symbol]["shares"]
                    old_avg_price = agent["positions"][symbol]["avg_price"]
                    total_cost = (old_shares * old_avg_price) + trade_value
                    total_shares = old_shares + shares
                    avg_price = total_cost / total_shares

                    agent["positions"][symbol] = {
                        "shares": total_shares,
                        "avg_price": avg_price,
                        "current_price": price,
                        "name": next(
                            (
                                s["name"]
                                for s in INDONESIAN_STOCKS
                                if s["symbol"] == symbol
                            ),
                            symbol,
                        ),
                    }
                else:
                    stock_name = next(
                        (s["name"] for s in INDONESIAN_STOCKS if s["symbol"] == symbol),
                        symbol,
                    )
                    agent["positions"][symbol] = {
                        "shares": shares,
                        "avg_price": price,
                        "current_price": price,
                        "name": stock_name,
                    }

                trade_record = {
                    "timestamp": time.time(),
                    "action": "BUY",
                    "symbol": symbol,
                    "shares": shares,
                    "price": price,
                    "value": trade_value,
                    "simulation_mode": simulation_mode,
                    "agent_id": agent_id,
                    "market_conditions": {
                        "volatility": market_cache.get(symbol, {}).get(
                            "change_percent", 0
                        )
                        / 100,
                        "trend": market_cache.get(symbol, {}).get("change_percent", 0)
                        / 100,
                        "volume": market_cache.get(symbol, {}).get("volume", 0),
                    },
                }
                agent["trades"].append(trade_record)
                agent["total_trades"] += 1

                # Record trade in analytics
                record_trade(agent_id, trade_record)

                if len(agent["trades"]) > 20:
                    agent["trades"] = agent["trades"][-20:]

                data_type = "REAL" if not simulation_mode else "SIMULASI"
                logger.info(
                    f"💰 {data_type} BELI - Agent {agent_id}: {shares} saham {symbol} @ Rp {price:,.2f} = Rp {trade_value:,.2f}"
                )
                return True
            else:
                logger.warning(
                    f"❌ Cash tidak cukup - Agent {agent_id}: Butuh Rp {trade_value:,.2f}, punya Rp {agent['cash']:,.2f}"
                )
                return False

        elif action == "SELL":
            # PERBAIKAN: Validasi posisi dengan lebih hati-hati
            if symbol not in agent["positions"]:
                logger.warning(
                    f"❌ Agent {agent_id} tidak punya posisi {symbol} - SKIP SELL"
                )
                return False

            if agent["positions"][symbol]["shares"] < shares:
                logger.warning(
                    f"❌ Agent {agent_id} hanya punya {agent['positions'][symbol]['shares']} saham {symbol}, ingin jual {shares} - SKIP"
                )
                return False

            sell_value = shares * price
            buy_value = shares * agent["positions"][symbol]["avg_price"]
            pnl = sell_value - buy_value

            agent["cash"] += sell_value

            agent["positions"][symbol]["shares"] -= shares

            if agent["positions"][symbol]["shares"] == 0:
                del agent["positions"][symbol]
            else:
                agent["positions"][symbol]["current_price"] = price

            agent["total_trades"] += 1
            if pnl > 0:
                agent["winning_trades"] += 1

            trade_record = {
                "timestamp": time.time(),
                "action": "SELL",
                "symbol": symbol,
                "shares": shares,
                "price": price,
                "value": sell_value,
                "pnl": pnl,
                "simulation_mode": simulation_mode,
                "agent_id": agent_id,
                "market_conditions": {
                    "volatility": market_cache.get(symbol, {}).get("change_percent", 0)
                    / 100,
                    "trend": market_cache.get(symbol, {}).get("change_percent", 0)
                    / 100,
                    "volume": market_cache.get(symbol, {}).get("volume", 0),
                },
            }
            agent["trades"].append(trade_record)

            # Record trade in analytics
            record_trade(agent_id, trade_record)

            if len(agent["trades"]) > 20:
                agent["trades"] = agent["trades"][-20:]

            data_type = "REAL" if not simulation_mode else "SIMULASI"
            logger.info(
                f"💰 {data_type} JUAL - Agent {agent_id}: {shares} saham {symbol} @ Rp {price:,.2f} = Rp {sell_value:,.2f} (P&L: Rp {pnl:,.2f})"
            )
            return True

        return False

    except Exception as e:
        logger.error(f"❌ Error mengeksekusi trade: {e}")
        return False


def calculate_portfolio_value(agent_id, market_data):
    """Calculate agent portfolio value"""
    try:
        agent = agents[agent_id]
        position_value = 0

        for symbol, position in agent["positions"].items():
            current_price = market_data.get(symbol, {}).get(
                "price", position.get("current_price", position["avg_price"])
            )
            position_value += position["shares"] * current_price

        total_cost = sum(
            pos["shares"] * pos["avg_price"] for pos in agent["positions"].values()
        )
        pnl = position_value - total_cost

        return {
            "cash": agent["cash"],
            "position_value": position_value,
            "total_value": agent["cash"] + position_value,
            "pnl": pnl,
            "pnl_percent": (pnl / 500000000) * 100,
            "position_count": len(agent["positions"]),
            "simulation_mode": simulation_mode,
        }

    except Exception as e:
        logger.error(f"❌ Error calculating portfolio value: {e}")
        return {
            "cash": agents[agent_id]["cash"],
            "position_value": 0,
            "total_value": agents[agent_id]["cash"],
            "pnl": 0,
            "pnl_percent": 0,
            "position_count": 0,
            "simulation_mode": simulation_mode,
        }


def simulate_ai_trading():
    """IMPROVED AI trading simulation - ALL agents should trade"""
    if not agents_active:
        return

    try:
        market_data_response = get_real_market_data()
        if not market_data_response["success"]:
            return

        market_data = market_data_response["data"]
        symbols = list(market_data.keys())

        if len(symbols) == 0:
            logger.warning("No market data available for AI trading")
            return

        # EACH agent has a chance to trade (increased from 30% to 60%)
        for agent_id in range(1, 6):
            if random.random() < 0.6:  # 60% chance per agent (increased from 30%)
                agent = agents[agent_id]
                strategy = agent["strategy"]

                # Strategy-based stock selection
                if strategy == "Conservative":
                    # Prefer large cap, stable stocks
                    preferred_stocks = [
                        s
                        for s in symbols
                        if any(x in s for x in ["BBCA", "BBRI", "TLKM", "UNVR"])
                    ]
                    if not preferred_stocks:
                        preferred_stocks = symbols[:10]  # Fallback to first 10 stocks
                elif strategy == "Aggressive":
                    # Prefer growth/volatile stocks
                    preferred_stocks = [
                        s
                        for s in symbols
                        if any(x in s for x in ["BKLA", "ITMG", "ADRO", "GIAA"])
                    ]
                    if not preferred_stocks:
                        preferred_stocks = symbols[-10:]  # Fallback to last 10 stocks
                elif strategy == "Balanced":
                    # Mix of everything
                    preferred_stocks = symbols
                elif strategy == "Trend":
                    # Prefer stocks with strong momentum
                    preferred_stocks = sorted(
                        symbols,
                        key=lambda s: abs(market_data[s]["change_percent"]),
                        reverse=True,
                    )[:10]
                else:  # Mean Reversion
                    # Prefer stocks that have moved significantly
                    preferred_stocks = [
                        s for s in symbols if abs(market_data[s]["change_percent"]) > 1
                    ]
                    if not preferred_stocks:
                        preferred_stocks = symbols

                # PERBAIKAN: Cek posisi yang dimiliki agent dulu
                owned_stocks = list(agent["positions"].keys())

                symbol = random.choice(
                    preferred_stocks if preferred_stocks else symbols
                )
                stock_data = market_data[symbol]

                # FIXED: Strategy-based decision making - NO continue statements!
                if strategy == "Conservative":
                    # Only trade on significant movements - ALWAYS make a decision
                    if stock_data["change_percent"] < -3:
                        action = "BUY"
                    elif stock_data["change_percent"] > 3:
                        action = "SELL"
                    else:
                        # FIXED: Instead of continue, make a balanced decision
                        action = random.choice(["BUY", "SELL"])

                # PERBAIKAN: Jika action SELL, pastikan agent punya posisi di saham tersebut
                if action == "SELL" and symbol not in owned_stocks:
                    # Ganti ke BUY atau cari saham lain yang dimiliki
                    if owned_stocks:
                        # 50% chance beli saham baru, 50% chance jual saham yang dimiliki
                        if random.random() < 0.5:
                            action = "BUY"
                        else:
                            symbol = random.choice(owned_stocks)
                            stock_data = market_data[symbol]
                    else:
                        action = "BUY"

                elif strategy == "Aggressive":
                    # Trade on smaller movements - ALWAYS make a decision
                    if stock_data["change_percent"] < -1:
                        action = "BUY"
                    elif stock_data["change_percent"] > 1:
                        action = "SELL"
                    else:
                        action = random.choice(["BUY", "SELL"])

                elif strategy == "Balanced":
                    # Balanced approach - ALWAYS make a decision
                    if stock_data["change_percent"] < -2:
                        action = "BUY"
                    elif stock_data["change_percent"] > 2.5:
                        action = "SELL"
                    else:
                        action = random.choice(["BUY", "SELL"])

                elif strategy == "Trend":
                    # Follow the trend - ALWAYS make a decision
                    if stock_data["change_percent"] > 0:
                        action = "BUY"
                    else:
                        action = "SELL"

                else:  # Mean Reversion
                    # Bet against the trend - ALWAYS make a decision
                    if stock_data["change_percent"] < -2:
                        action = "SELL"  # Sell on further drops
                    elif stock_data["change_percent"] > 3:
                        action = "BUY"  # Buy on further gains
                    else:
                        action = random.choice(["BUY", "SELL"])

                # Calculate position size based on strategy and agent risk
                if strategy == "Conservative":
                    max_position_value = 50000000  # 50M max position
                    shares = min(100, int(max_position_value / stock_data["price"]))
                elif strategy == "Aggressive":
                    max_position_value = 200000000  # 200M max position
                    shares = min(500, int(max_position_value / stock_data["price"]))
                else:
                    max_position_value = 100000000  # 100M max position
                    shares = min(300, int(max_position_value / stock_data["price"]))

                # Ensure we always have at least some shares to trade
                if shares < 1:
                    shares = 10

                if shares > 0:
                    price = stock_data["price"]
                    success = execute_trade(agent_id, symbol, action, shares, price)

                    if success:
                        agents[agent_id]["status"] = "trading"
                        data_type = "REAL" if not simulation_mode else "SIMULASI"
                        agents[agent_id]["last_action"] = (
                            f"{data_type} {action} {symbol}"
                        )

                        logger.info(
                            f"🤖 AI Agent {agent_id} ({strategy}) berhasil {data_type} {action} {shares} saham {symbol} @ Rp {price:,.2f}"
                        )
                    else:
                        agents[agent_id]["status"] = "active"
                        agents[agent_id]["last_action"] = f"Gagal {action} {symbol}"

    except Exception as e:
        logger.error(f"❌ Error in AI trading simulation: {e}")


def get_market_status():
    """Get current Indonesian market status"""
    now = datetime.now()
    wib_time = now + timedelta(hours=7)

    market_open = wib_time.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = wib_time.replace(hour=15, minute=30, second=0, microsecond=0)

    if now.weekday() >= 5:
        return "TUTUP - AKHIR PEKAN"
    elif wib_time < market_open:
        return "PRA-MARKET"
    elif wib_time > market_close:
        return "TUTUP"
    else:
        return "TRADING AKTIF"


# IMPROVED HTML Template with better colors and visibility
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>📈 SISTEM TRADING SAHAM INDONESIA - DIPERBAIKI</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 10px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.98);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 {
            color: #1e3c72;
            text-align: center;
            margin-bottom: 20px;
            font-size: 32px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            font-weight: bold;
        }
        .status {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        .status h3 {
            font-size: 20px;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .status p {
            font-size: 16px;
            margin: 5px 0;
        }
        .market-status {
            background: linear-gradient(135deg, #2196f3, #1976d2);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            text-align: center;
        }
        .market-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .summary-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #2196f3;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .summary-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .summary-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .control-panel {
            background: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }
        button {
            background: #4CAF50;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            margin: 5px;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.3s;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        button:hover {
            background: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .red { background: #f44336; }
        .red:hover { background: #da190b; }
        .blue { background: #2196f3; }
        .blue:hover { background: #0b7dda; }
        .orange { background: #ff9800; }
        .orange:hover { background: #e68900; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .agent {
            background: #fff;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: all 0.3s;
            position: relative;
            border: 2px solid #ddd;
        }
        .agent:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }
        .agent.active { border-left: 5px solid #4CAF50; border-color: #4CAF50; }
        .agent.inactive { border-left: 5px solid #f44336; border-color: #f44336; }
        .agent.trading { border-left: 5px solid #2196f3; border-color: #2196f3; }
        .agent h4 {
            color: #1e3c72;
            margin-bottom: 10px;
            font-size: 18px;
            font-weight: bold;
        }
        .agent-strategy {
            color: #666;
            font-size: 12px;
            margin-bottom: 15px;
            font-style: italic;
        }
        .status-indicator {
            position: absolute;
            top: 15px;
            right: 15px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-active { background: #4CAF50; }
        .status-inactive { background: #f44336; }
        .status-trading { background: #2196f3; }
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
            100% { opacity: 1; transform: scale(1); }
        }
        .capital-info {
            background: #e8f5e8;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .capital-row {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            font-size: 14px;
        }
        .capital-label {
            color: #333;
            font-weight: 500;
        }
        .capital-value {
            font-weight: bold;
            color: #1e3c72;
        }
        .positive { color: #2e7d32 !important; }
        .negative { color: #c62828 !important; }
        .neutral { color: #666 !important; }
        .positions {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            max-height: 150px;
            overflow-y: auto;
        }
        .position-item {
            background: #fff;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            font-size: 12px;
            border-left: 3px solid #2196f3;
        }
        .position-item strong {
            color: #1e3c72;
        }
        .activity-log {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            max-height: 100px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 11px;
            color: #333;
        }
        .market-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stock-card {
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .stock-card:hover {
            border-color: #2196f3;
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(33,150,243,0.2);
        }
        .stock-symbol {
            font-weight: bold;
            font-size: 14px;
            color: #1e3c72;
            margin-bottom: 5px;
        }
        .stock-name {
            font-size: 10px;
            color: #666;
            margin-bottom: 8px;
        }
        .stock-sector {
            font-size: 9px;
            color: #999;
            margin-bottom: 8px;
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
        }
        .stock-price {
            font-size: 16px;
            font-weight: bold;
            margin: 5px 0;
            color: #333;
        }
        .stock-change {
            font-size: 11px;
            padding: 4px 8px;
            border-radius: 5px;
            margin: 5px 0;
            font-weight: bold;
        }
        .loading {
            color: #333;
            font-style: italic;
            text-align: center;
            padding: 20px;
        }
        .error {
            color: #f44336;
            background: #ffebee;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            font-weight: 500;
        }
        .success {
            color: #2e7d32;
            background: #e8f5e8;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            font-weight: 500;
        }
        .warning {
            color: #ff9800;
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            font-weight: 500;
        }
        .info {
            text-align: center;
            color: #333;
            margin: 20px 0;
            font-size: 12px;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
        }
        .data-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .data-indicator.simulation {
            background: #ff9800;
        }
        .data-indicator.error {
            background: #f44336;
        }
        h3 {
            color: #1e3c72;
            font-weight: bold;
        }
        .last-update {
            font-size: 11px;
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="data-indicator" id="dataIndicator">🔄 MEMUAT...</div>

    <div class="container">
        <h1>📈 SISTEM TRADING SAHAM INDONESIA - VERSI DIPERBAIKI</h1>

        <div class="status">
            <h3>🇮🇩 BURSA SAHAM INDONESIA (IDX) - SISTEM HYBRID</h3>
            <p><strong>Mode:</strong> <span id="systemMode">MEMUAT...</span></p>
            <p><strong>Status Market:</strong> <span id="marketStatus">MENGHUBUNGKAN...</span></p>
            <p><strong>Total Modal:</strong> Rp 2.500.000.000 | <strong>AI Agents:</strong> 5 Trader AI</p>
            <p><strong>Update Terakhir:</strong> <span id="lastUpdate">Loading...</span></p>
        </div>

        <div class="market-summary">
            <div class="summary-card">
                <div class="summary-value" id="currentCapital">Rp 2.500.000.000</div>
                <div class="summary-label">Nilai Portfolio Saat Ini</div>
            </div>
            <div class="summary-card">
                <div class="summary-value" id="totalPnL">Rp 0</div>
                <div class="summary-label">Total P&L</div>
            </div>
            <div class="summary-card">
                <div class="summary-value" id="totalPnLPerc">0.00%</div>
                <div class="summary-label">Persentase P&L</div>
            </div>
            <div class="summary-card">
                <div class="summary-value" id="totalTrades">0</div>
                <div class="summary-label">Total Transaksi</div>
            </div>
            <div class="summary-card">
                <div class="summary-value" id="winRate">0.00%</div>
                <div class="summary-label">Win Rate</div>
            </div>
            <div class="summary-card">
                <div class="summary-value" id="activePositions">0</div>
                <div class="summary-label">Posisi Aktif</div>
            </div>
        </div>

        <div class="control-panel">
            <button id="startBtn" onclick="startAgents()">🚀 Mulai Trading</button>
            <button id="stopBtn" onclick="stopAgents()" disabled>⏹️ Stop Trading</button>
            <button class="blue" onclick="refreshMarketData()">🔄 Refresh Data</button>
            <button class="orange" onclick="resetAll()">🔄 Reset Semua</button>
            <button class="red" onclick="switchDataMode()">🔄 Ganti Mode Data</button>
        </div>

        <div id="marketSection">
            <h3>📈 Data Market (<span id="dataSource">Loading...</span>)</h3>
            <div class="market-grid" id="marketGrid">
                <div class="loading">Memuat data market...</div>
            </div>
        </div>

        <h3>🤖 AI Trading Agents</h3>
        <div class="grid" id="agentsGrid">
            <!-- Agents will be populated here -->
        </div>

        <div class="info">
            <p>📊 Sistem Hybrid: Real Yahoo Finance API + Simulasi Realistis</p>
            <p>💰 Paper Trading (Tidak Uang Asli) | 🎯 Bursa Sahham Indonesia (IDX)</p>
            <p>🔴 Oranye = Mode Simulasi | 🟢 Hijau = Mode API Real</p>
            <p class="last-update">Auto-refresh setiap 30 detik | Klik saham apapun untuk eksekusi trade</p>
        </div>
    </div>

    <script>
        let agentsActive = false;
        let marketData = {};
        let marketUpdateInterval;
        let aiTradingInterval;
        let currentSimulationMode = true;

        const agentInfo = [
            { id: 1, name: 'Conservative Agent', strategy: 'Low-risk, blue chips', color: '#4CAF50' },
            { id: 2, name: 'Aggressive Agent', strategy: 'High-growth stocks', color: '#f44336' },
            { id: 3, name: 'Balanced Agent', strategy: 'Diversified portfolio', color: '#2196f3' },
            { id: 4, name: 'Trend Agent', strategy: 'Momentum trading', color: '#ff9800' },
            { id: 5, name: 'Mean Reversion', strategy: 'Contrarian trading', color: '#9c27b0' }
        ];

        function formatCurrency(amount) {
            return 'Rp ' + Math.round(amount).toLocaleString('id-ID');
        }

        function initializeAgents() {
            const grid = document.getElementById('agentsGrid');
            grid.innerHTML = agentInfo.map(agent => `
                <div class="agent inactive" id="agent${agent.id}">
                    <div class="status-indicator status-inactive" id="status${agent.id}"></div>
                    <h4 style="color: ${agent.color};">${agent.name}</h4>
                    <p class="agent-strategy">${agent.strategy}</p>
                    <div class="capital-info">
                        <div class="capital-row">
                            <span class="capital-label">Cash:</span>
                            <span class="capital-value" id="cash${agent.id}">Rp 500.000.000</span>
                        </div>
                        <div class="capital-row">
                            <span class="capital-label">Nilai Posisi:</span>
                            <span class="capital-value" id="positionValue${agent.id}">Rp 0</span>
                        </div>
                        <div class="capital-row">
                            <span class="capital-label">Total Nilai:</span>
                            <span class="capital-value" id="totalValue${agent.id}">Rp 500.000.000</span>
                        </div>
                        <div class="capital-row">
                            <span class="capital-label">P&L:</span>
                            <span class="capital-value neutral" id="pnl${agent.id}">Rp 0</span>
                        </div>
                        <div class="capital-row">
                            <span class="capital-label">Status:</span>
                            <span class="capital-value" id="agentStatus${agent.id}">IDLE</span>
                        </div>
                    </div>
                    <div class="positions" id="positions${agent.id}">
                        <div style="color: #666; text-align: center;">Tidak ada posisi terbuka</div>
                    </div>
                    <div class="activity-log" id="log${agent.id}">
                        <div>[${new Date().toLocaleTimeString()}] Agent diinisialisasi dengan modal Rp 500M - SIAP</div>
                    </div>
                </div>
            `).join('');
        }

        function updateDataIndicator() {
            // PERBAIKAN: Tambahkan validasi untuk semua elemen
            const indicator = document.getElementById('dataIndicator');
            const systemMode = document.getElementById('systemMode');
            const dataSource = document.getElementById('dataSource');

            if (currentSimulationMode) {
                if (indicator) {
                    indicator.textContent = '🟠 MODE SIMULASI';
                    indicator.className = 'data-indicator simulation';
                }
                if (systemMode) systemMode.textContent = 'SIMULASI MARKET REALISTIS';
                if (dataSource) dataSource.textContent = 'Simulasi Market Realistis';
            } else {
                if (indicator) {
                    indicator.textContent = '🟢 MODE API REAL';
                    indicator.className = 'data-indicator';
                }
                if (systemMode) systemMode.textContent = 'API YAHOO FINANCE REAL';
                if (dataSource) dataSource.textContent = 'Yahoo Finance API';
            }
        }

        function fetchMarketData() {
            fetch('/api/market-data')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        marketData = data.data;
                        currentSimulationMode = data.simulation_mode;
                        updateDataIndicator();
                        updateMarketDisplay();
                        updateAllAgents();

                        // PERBAIKAN: Tambahkan validasi untuk elemen
                        const lastUpdateElement = document.getElementById('lastUpdate');
                        const marketStatusElement = document.getElementById('marketStatus');
                        if (lastUpdateElement) lastUpdateElement.textContent = new Date().toLocaleString();
                        if (marketStatusElement) marketStatusElement.textContent = 'TERHUBUNG';

                        fetch('/api/market-status')
                            .then(response => response.json())
                            .then(statusData => {
                                const marketStatusDetailElement = document.getElementById('marketStatusDetail');
                                if (marketStatusDetailElement) marketStatusDetailElement.textContent = statusData.status;
                            });

                        if (!data.cached) {
                            if (data.simulation_mode) {
                                showInfo('⚠️ Yahoo Finance API tidak tersedia - Menggunakan simulasi market realistis');
                            } else {
                                showSuccess('✅ Terhubung ke Yahoo Finance API real!');
                            }
                        }
                    } else {
                        showError('Gagal mengambil data market: ' + data.error);
                        document.getElementById('dataIndicator').textContent = '🔴 ERROR';
                        document.getElementById('dataIndicator').className = 'data-indicator error';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    showError('Koneksi error: ' + error.message);
                    document.getElementById('dataIndicator').textContent = '🔴 ERROR';
                    document.getElementById('dataIndicator').className = 'data-indicator error';
                });
        }

        function updateMarketDisplay() {
            const grid = document.getElementById('marketGrid');

            if (!marketData || Object.keys(marketData).length === 0) {
                grid.innerHTML = '<div class="error">Tidak ada data market tersedia</div>';
                return;
            }

            const stocks = Object.entries(marketData)
                .sort((a, b) => Math.abs(b[1].change_percent) - Math.abs(a[1].change_percent))
                .slice(0, 20);

            grid.innerHTML = stocks.map(([symbol, stock]) => {
                const changeClass = stock.change_percent > 0 ? 'positive' : stock.change_percent < 0 ? 'negative' : 'neutral';
                const changeSymbol = stock.change_percent >= 0 ? '+' : '';

                return `
                    <div class="stock-card" onclick="executeTrade('${symbol}')" title="Klik untuk trading ${stock.name}">
                        <div class="stock-symbol">${symbol.replace('.JK', '')}</div>
                        <div class="stock-name">${stock.name}</div>
                        <div class="stock-sector">${stock.sector}</div>
                        <div class="stock-price">${formatCurrency(stock.price)}</div>
                        <div class="stock-change ${changeClass}">${changeSymbol}${stock.change_percent.toFixed(2)}%</div>
                        <div style="font-size: 10px; color: #666;">Vol: ${(stock.volume/1000000).toFixed(1)}M</div>
                    </div>
                `;
            }).join('');
        }

        function updateAllAgents() {
            let totalValue = 0;
            let totalPnL = 0;
            let totalTrades = 0;
            let totalWinningTrades = 0;
            let totalPositions = 0;

            for (let agent of agentInfo) {
                fetch(`/api/agent/${agent.id}/status`)
                    .then(response => response.json())
                    .then(data => {
                        // PERBAIKAN: Tambahkan validasi sebelum mengakses elemen
                        const cashElement = document.getElementById(`cash${agent.id}`);
                        const positionValueElement = document.getElementById(`positionValue${agent.id}`);
                        const totalValueElement = document.getElementById(`totalValue${agent.id}`);
                        const agentStatusElement = document.getElementById(`agentStatus${agent.id}`);
                        const pnlElement = document.getElementById(`pnl${agent.id}`);
                        const positionsDiv = document.getElementById(`positions${agent.id}`);

                        if (cashElement) cashElement.textContent = formatCurrency(data.cash);
                        if (positionValueElement) positionValueElement.textContent = formatCurrency(data.position_value);
                        if (totalValueElement) totalValueElement.textContent = formatCurrency(data.total_value);
                        if (agentStatusElement) agentStatusElement.textContent = data.status.toUpperCase();

                        if (pnlElement) {
                            pnlElement.textContent = formatCurrency(data.pnl);
                            pnlElement.className = `capital-value ${data.pnl >= 0 ? 'positive' : 'negative'}`;
                        }

                        if (positionsDiv) {
                            if (data.positions && Object.keys(data.positions).length > 0) {
                                positionsDiv.innerHTML = '<div style="font-weight: bold; margin-bottom: 8px; color: #1e3c72;">Posisi Saat Ini:</div>' +
                                    Object.entries(data.positions).map(([symbol, pos]) => {
                                        const currentPrice = marketData[symbol]?.price || pos.current_price;
                                        const pnl = (currentPrice - pos.avg_price) * pos.shares;
                                        return `
                                            <div class="position-item">
                                                <div><strong>${symbol.replace('.JK', '')}</strong> - ${pos.name}</div>
                                                <div>${pos.shares} saham @ ${formatCurrency(pos.avg_price)}</div>
                                                <div>Saat ini: ${formatCurrency(currentPrice)} | P&L: <span class="${pnl >= 0 ? 'positive' : 'negative'}">${formatCurrency(pnl)}</span></div>
                                            </div>
                                        `;
                                    }).join('');
                                totalPositions += Object.keys(data.positions).length;
                            } else {
                                positionsDiv.innerHTML = '<div style="color: #666; text-align: center;">Tidak ada posisi terbuka</div>';
                            }
                        }

                        totalValue += data.total_value;
                        totalPnL += data.pnl;
                        totalTrades += data.total_trades;
                        totalWinningTrades += data.winning_trades;

                        updateSummary(totalValue, totalPnL, totalTrades, totalWinningTrades, totalPositions);
                    })
                    .catch(error => {
                        console.error(`Error updating agent ${agent.id}:`, error);
                    });
            }
        }

        function updateSummary(totalValue, totalPnL, totalTrades, totalWinningTrades, totalPositions) {
            const initialCapital = 2500000000;
            const pnlPercentage = (totalPnL / initialCapital) * 100;
            const winRate = totalTrades > 0 ? (totalWinningTrades / totalTrades * 100) : 0;

            // PERBAIKAN: Tambahkan validasi untuk semua elemen
            const currentCapitalElement = document.getElementById('currentCapital');
            const totalPnLElement = document.getElementById('totalPnL');
            const totalPnLPercElement = document.getElementById('totalPnLPerc');
            const totalTradesElement = document.getElementById('totalTrades');
            const winRateElement = document.getElementById('winRate');
            const activePositionsElement = document.getElementById('activePositions');

            if (currentCapitalElement) currentCapitalElement.textContent = formatCurrency(totalValue);
            if (totalPnLElement) totalPnLElement.textContent = formatCurrency(totalPnL);
            if (totalPnLPercElement) totalPnLPercElement.textContent = pnlPercentage.toFixed(2) + '%';
            if (totalTradesElement) totalTradesElement.textContent = totalTrades;
            if (winRateElement) winRateElement.textContent = winRate.toFixed(2) + '%';
            if (activePositionsElement) activePositionsElement.textContent = totalPositions;

            const pnlElement = document.getElementById('totalPnL');
            const pnlPercElement = document.getElementById('totalPnLPerc');
            if (totalPnL >= 0) {
                if (pnlElement) pnlElement.className = 'summary-value positive';
                if (pnlPercElement) pnlPercElement.className = 'summary-value positive';
            } else {
                if (pnlElement) pnlElement.className = 'summary-value negative';
                if (pnlPercElement) pnlPercElement.className = 'summary-value negative';
            }
        }

        function startAgents() {
            if (Object.keys(marketData).length === 0) {
                showError('Silakan refresh data market terlebih dahulu!');
                return;
            }

            agentsActive = true;
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;

            fetch('/api/start-agents', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showSuccess('🚀 Semua AI Trading Agent dimulai!');
                        updateAgentStatuses('active');
                        startAITrading();
                    } else {
                        showError('Gagal memulai agents: ' + data.error);
                    }
                });
        }

        function stopAgents() {
            agentsActive = false;
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;

            fetch('/api/stop-agents', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showSuccess('⏹️ Semua AI Trading Agent dihentikan!');
                        updateAgentStatuses('inactive');
                    }
                });

            clearInterval(aiTradingInterval);
        }

        function startAITrading() {
            aiTradingInterval = setInterval(() => {
                if (agentsActive && Object.keys(marketData).length > 0) {
                    simulateAITrade();
                }
            }, 5000); // Every 5 seconds for more activity
        }

        function simulateAITrade() {
            const agentId = Math.floor(Math.random() * 5) + 1;
            const symbols = Object.keys(marketData);

            if (symbols.length === 0) return;

            const symbol = symbols[Math.floor(Math.random() * symbols.length)];
            const stockData = marketData[symbol];
            const action = Math.random() > 0.5 ? 'BUY' : 'SELL';
            const shares = Math.floor(Math.random() * 100) + 10;
            const price = stockData.price;

            fetch('/api/execute-trade', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({agent_id: agentId, symbol: symbol, action: action, shares: shares})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateAllAgents();
                    const logDiv = document.getElementById(`log${agentId}`);
                    const newLog = `[${new Date().toLocaleTimeString()}] ${action} ${shares} ${symbol} @ ${formatCurrency(price)}`;
                    logDiv.innerHTML = '<div>' + newLog + '</div>' + logDiv.innerHTML;

                    const logs = logDiv.children;
                    if (logs.length > 5) {
                        logDiv.removeChild(logs[logs.length - 1]);
                    }
                }
            });
        }

        function executeTrade(symbol) {
            if (!agentsActive) {
                showError('Silakan mulai agents terlebih dahulu!');
                return;
            }

            const agentId = Math.floor(Math.random() * 5) + 1;
            const action = Math.random() > 0.5 ? 'BUY' : 'SELL';
            const shares = Math.floor(Math.random() * 100) + 10;
            const price = marketData[symbol].price;

            fetch('/api/execute-trade', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({agent_id: agentId, symbol: symbol, action: action, shares: shares})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateAllAgents();
                    showSuccess(`Trade dieksekusi: ${action} ${shares} ${symbol} @ ${formatCurrency(price)}`);
                } else {
                    showError('Trade gagal: ' + data.error);
                }
            });
        }

        function refreshMarketData() {
            fetchMarketData();
        }

        function resetAll() {
            if (confirm('Apakah Anda yakin ingin mereset semua agents dan transaksi?')) {
                fetch('/api/reset-all', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            showSuccess('Sistem berhasil direset!');
                            initializeAgents();
                            updateAllAgents();
                        }
                    });
            }
        }

        function switchDataMode() {
            fetch('/api/switch-data-mode', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showSuccess('Mode data berhasil diganti!');
                        fetchMarketData();
                    }
                });
        }

        function updateAgentStatuses(status) {
            for (let agent of agentInfo) {
                const agentDiv = document.getElementById(`agent${agent.id}`);
                const statusDiv = document.getElementById(`status${agent.id}`);

                agentDiv.className = `agent ${status}`;
                statusDiv.className = `status-indicator status-${status}`;
            }
        }

        function showSuccess(message) {
            showNotification(message, 'success');
        }

        function showError(message) {
            showNotification(message, 'error');
        }

        function showInfo(message) {
            showNotification(message, 'info');
        }

        function showNotification(message, type) {
            const notification = document.createElement('div');
            notification.className = type;
            notification.innerHTML = message;
            notification.style.cssText = 'position: fixed; top: 100px; left: 50%; transform: translateX(-50%); z-index: 9999; padding: 15px 20px; border-radius: 8px; font-weight: bold; color: #333;';

            document.body.appendChild(notification);

            setTimeout(() => {
                document.body.removeChild(notification);
            }, 5000);
        }

        // Initialize
        initializeAgents();
        fetchMarketData();

        // Auto-refresh data
        marketUpdateInterval = setInterval(fetchMarketData, 30000);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/market-data")
def api_market_data():
    return jsonify(get_real_market_data())


@app.route("/api/market-status")
def api_market_status():
    return jsonify({"status": get_market_status()})


@app.route("/api/agent/<int:agent_id>/status")
def api_agent_status(agent_id):
    market_data_response = get_real_market_data()
    if market_data_response["success"]:
        portfolio = calculate_portfolio_value(agent_id, market_data_response["data"])
        portfolio.update(agents[agent_id])
        return jsonify(portfolio)
    else:
        return jsonify({"error": "Market data unavailable"})


@app.route("/api/start-agents", methods=["POST"])
def api_start_agents():
    global agents_active, simulation_mode
    agents_active = True

    if try_yahoo_finance_connection():
        simulation_mode = False
        logger.info("Beralih ke mode API Yahoo Finance REAL")

    # Start analytics session
    start_trading_session()

    for agent_id in agents:
        agents[agent_id]["status"] = "active"
        agents[agent_id]["last_action"] = "Memulai trading"

    return jsonify(
        {"success": True, "mode": "REAL" if not simulation_mode else "SIMULATION"}
    )


@app.route("/api/stop-agents", methods=["POST"])
def api_stop_agents():
    global agents_active
    agents_active = False

    for agent_id in agents:
        agents[agent_id]["status"] = "inactive"
        agents[agent_id]["last_action"] = "Berhenti trading"

    return jsonify({"success": True})


@app.route("/api/execute-trade", methods=["POST"])
def api_execute_trade():
    try:
        data = request.get_json()
        agent_id = data["agent_id"]
        symbol = data["symbol"]
        action = data["action"]
        shares = data["shares"]

        market_data_response = get_real_market_data()
        if market_data_response["success"]:
            price = market_data_response["data"][symbol]["price"]
            success = execute_trade(agent_id, symbol, action, shares, price)

            if success:
                return jsonify({"success": True})
            else:
                return jsonify({"success": False, "error": "Trade execution failed"})
        else:
            return jsonify({"success": False, "error": "Market data unavailable"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/reset-all", methods=["POST"])
def api_reset_all():
    global agents_active, market_cache, price_history
    agents_active = False

    # Reset agents
    for agent_id in agents:
        agents[agent_id] = {
            "cash": 500000000,
            "positions": {},
            "trades": [],
            "total_trades": 0,
            "winning_trades": 0,
            "status": "idle",
            "last_action": "Menunggu data market...",
            "strategy": agents[agent_id]["strategy"],
            "color": agents[agent_id]["color"],
        }

    # Reset price history
    for stock in INDONESIAN_STOCKS:
        price_history[stock["symbol"]] = {
            "prices": [stock["base_price"]],
            "trend": random.choice([-0.01, 0, 0.01]),
            "volatility": random.uniform(0.02, 0.08),
        }

    # Clear cache
    market_cache = {}

    return jsonify({"success": True})


@app.route("/api/switch-data-mode", methods=["POST"])
def api_switch_data_mode():
    global simulation_mode
    simulation_mode = not simulation_mode

    global market_cache
    market_cache = {}

    logger.info(f"Beralih ke mode {'API REAL' if not simulation_mode else 'SIMULASI'}")

    return jsonify(
        {"success": True, "new_mode": "REAL" if not simulation_mode else "SIMULATION"}
    )


# Analytics API endpoints
@app.route("/api/analytics/performance")
def api_analytics_performance():
    """Get performance analytics data"""
    try:
        days_back = request.args.get("days", 7, type=int)

        with sqlite3.connect("analytics.db") as conn:
            cursor = conn.cursor()

            # Get performance data
            cursor.execute(
                """
                SELECT agent_id, strategy, SUM(pnl) as total_pnl,
                       COUNT(*) as trade_count, AVG(pnl) as avg_pnl
                FROM individual_trades
                WHERE created_at >= date('now', '-{} days')
                GROUP BY agent_id, strategy
                ORDER BY total_pnl DESC
            """.format(days_back)
            )

            performance_data = []
            for row in cursor.fetchall():
                performance_data.append(
                    {
                        "agent_id": row[0],
                        "strategy": row[1],
                        "total_pnl": row[2],
                        "trade_count": row[3],
                        "avg_pnl": row[4],
                    }
                )

            return jsonify(
                {"success": True, "data": performance_data, "period_days": days_back}
            )

    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analytics/signals")
def api_analytics_signals():
    """Get trading signals based on analytics"""
    try:
        signals = get_market_signals()

        return jsonify({"success": True, "data": signals})

    except Exception as e:
        logger.error(f"Error getting trading signals: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analytics/report")
def api_analytics_report():
    """Get comprehensive analytics report"""
    try:
        days_back = request.args.get("days", 7, type=int)

        with sqlite3.connect("analytics.db") as conn:
            cursor = conn.cursor()

            # Get basic stats
            cursor.execute(
                """
                SELECT
                    COUNT(DISTINCT session_id) as sessions,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM individual_trades
                WHERE created_at >= date('now', '-{} days')
            """.format(days_back)
            )

            stats = cursor.fetchone()

            # Get strategy performance
            cursor.execute(
                """
                SELECT strategy, COUNT(*) as trades, SUM(pnl) as pnl
                FROM individual_trades
                WHERE created_at >= date('now', '-{} days')
                GROUP BY strategy
                ORDER BY pnl DESC
            """.format(days_back)
            )

            strategy_data = []
            for row in cursor.fetchall():
                strategy_data.append(
                    {"strategy": row[0], "trades": row[1], "pnl": row[2]}
                )

            win_rate = (stats[2] / stats[1] * 100) if stats[1] > 0 else 0

            report = {
                "period_days": days_back,
                "sessions": stats[0],
                "total_trades": stats[1],
                "winning_trades": stats[2],
                "win_rate": round(win_rate, 2),
                "total_pnl": stats[3] or 0,
                "avg_pnl": stats[4] or 0,
                "best_trade": stats[5] or 0,
                "worst_trade": stats[6] or 0,
                "strategy_performance": strategy_data,
            }

            return jsonify({"success": True, "data": report})

    except Exception as e:
        logger.error(f"Error getting analytics report: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    logger.info("🚀 Starting IMPROVED Indonesian Stock Market AI Trading System")
    logger.info("📡 Testing Yahoo Finance API connection...")

    if try_yahoo_finance_connection():
        simulation_mode = False
        logger.info("✅ Yahoo Finance API tersedia - Starting in REAL mode")
    else:
        logger.info("⚠️ Yahoo Finance API tidak tersedia - Starting in SIMULATION mode")

    logger.info("🌐 Web interface tersedia di: http://localhost:5002")
    logger.info("🤖 5 AI Agents dengan Rp 500M masing-masing (Total: Rp 2.5 Miliar)")
    logger.info(
        "🔧 Perbaikan: Semua agents sekarang bisa trading & dashboard lebih mudah dibaca"
    )

    app.run(host="0.0.0.0", port=5002, debug=True)
