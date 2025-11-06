import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MarketDataService:
    def __init__(self):
        self.cache = {}
        self.cache_duration = 60  # seconds

    def get_real_time_data(self, symbol):
        """Get real-time market data for a symbol"""
        try:
            # Check cache first
            if symbol in self.cache:
                cached_data, timestamp = self.cache[symbol]
                if (datetime.now() - timestamp).seconds < self.cache_duration:
                    return cached_data

            # Fetch from Yahoo Finance
            ticker = yf.Ticker(symbol)
            info = ticker.info

            data = {
                "symbol": symbol,
                "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                "volume": info.get("volume", 0),
                "open": info.get("regularMarketOpen", 0),
                "high": info.get("regularMarketDayHigh", 0),
                "low": info.get("regularMarketDayLow", 0),
                "previous_close": info.get("regularMarketPreviousClose", 0),
                "timestamp": datetime.now(),
            }

            # Cache the data
            self.cache[symbol] = (data, datetime.now())

            return data

        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol, period="1y"):
        """Get historical data for technical analysis"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return None

    def calculate_technical_indicators(self, symbol, period="1y"):
        """Calculate technical indicators for a symbol"""
        try:
            hist_data = self.get_historical_data(symbol, period)
            if hist_data is None or hist_data.empty:
                return None

            # Calculate RSI
            rsi = ta.momentum.rsi(hist_data["Close"], window=14)

            # Calculate MACD
            macd_line = ta.trend.macd(hist_data["Close"])
            macd_signal = ta.trend.macd_signal(hist_data["Close"])

            # Calculate Bollinger Bands
            bollinger_upper = ta.volatility.bollinger_hband(hist_data["Close"])
            bollinger_lower = ta.volatility.bollinger_lband(hist_data["Close"])
            bollinger_middle = ta.volatility.bollinger_mavg(hist_data["Close"])

            # Calculate ATR
            atr = ta.volatility.average_true_range(
                hist_data["High"], hist_data["Low"], hist_data["Close"]
            )

            # Current values
            current_price = hist_data["Close"].iloc[-1]

            indicators = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "price": current_price,
                "rsi": rsi.iloc[-1] if not rsi.empty else None,
                "macd": macd_line.iloc[-1] if not macd_line.empty else None,
                "macd_signal": macd_signal.iloc[-1] if not macd_signal.empty else None,
                "bollinger_upper": bollinger_upper.iloc[-1]
                if not bollinger_upper.empty
                else None,
                "bollinger_lower": bollinger_lower.iloc[-1]
                if not bollinger_lower.empty
                else None,
                "bollinger_middle": bollinger_middle.iloc[-1]
                if not bollinger_middle.empty
                else None,
                "atr": atr.iloc[-1] if not atr.empty else None,
                "volume_ratio": hist_data["Volume"].iloc[-1]
                / hist_data["Volume"].mean()
                if not hist_data["Volume"].empty
                else 1.0,
            }

            return indicators

        except Exception as e:
            logger.error(f"Failed to calculate indicators for {symbol}: {e}")
            return None

    def store_market_data(self, symbol, data):
        """Store market data to database"""
        try:
            from database.connection import DatabaseConnection

            db = DatabaseConnection()

            query = """
                INSERT INTO market_data
                (symbol, timestamp, open_price, high_price, low_price, close_price, volume,
                 rsi, macd, bollinger_upper, bollinger_lower)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            params = (
                symbol,
                data["timestamp"],
                data.get("open"),
                data.get("high"),
                data.get("low"),
                data.get("price"),
                data.get("volume"),
                data.get("rsi"),
                data.get("macd"),
                data.get("bollinger_upper"),
                data.get("bollinger_lower"),
            )

            db.execute(query, params)
            logger.info(f"Stored market data for {symbol}")

        except Exception as e:
            logger.error(f"Failed to store market data for {symbol}: {e}")
