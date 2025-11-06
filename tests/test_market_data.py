import pytest
from unittest.mock import Mock, patch
import pandas as pd
from services.market_data import MarketDataService


def test_get_real_time_data():
    service = MarketDataService()

    # Mock the yfinance Ticker to avoid API rate limits
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            "currentPrice": 150.25,
            "volume": 1000000,
            "regularMarketOpen": 149.50,
            "regularMarketDayHigh": 151.00,
            "regularMarketDayLow": 148.75,
            "regularMarketPreviousClose": 149.00,
        }
        mock_ticker.return_value = mock_ticker_instance

        data = service.get_real_time_data("AAPL")

        assert data is not None
        assert "price" in data
        assert "volume" in data
        assert "timestamp" in data
        assert data["price"] == 150.25
        assert data["volume"] == 1000000
        assert data["symbol"] == "AAPL"


def test_get_real_time_data_api_failure():
    service = MarketDataService()

    # Test API failure handling
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.side_effect = Exception("API Error")

        data = service.get_real_time_data("AAPL")

        assert data is None


def test_calculate_technical_indicators():
    service = MarketDataService()

    # Mock yfinance historical data
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker_instance = Mock()

        # Create mock historical data
        import pandas as pd

        mock_hist = pd.DataFrame(
            {
                "Open": [
                    145,
                    146,
                    147,
                    148,
                    149,
                    150,
                    151,
                    152,
                    153,
                    154,
                    155,
                    156,
                    157,
                    158,
                    159,
                ],
                "High": [
                    146,
                    147,
                    148,
                    149,
                    150,
                    151,
                    152,
                    153,
                    154,
                    155,
                    156,
                    157,
                    158,
                    159,
                    160,
                ],
                "Low": [
                    144,
                    145,
                    146,
                    147,
                    148,
                    149,
                    150,
                    151,
                    152,
                    153,
                    154,
                    155,
                    156,
                    157,
                    158,
                ],
                "Close": [
                    145.5,
                    146.5,
                    147.5,
                    148.5,
                    149.5,
                    150.5,
                    151.5,
                    152.5,
                    153.5,
                    154.5,
                    155.5,
                    156.5,
                    157.5,
                    158.5,
                    159.5,
                ],
                "Volume": [1000000] * 15,
            }
        )

        mock_ticker_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_ticker_instance

        indicators = service.calculate_technical_indicators("AAPL")

        assert indicators is not None
        assert "rsi" in indicators
        assert "macd" in indicators
        assert "bollinger_upper" in indicators
        assert "bollinger_lower" in indicators
        assert "symbol" in indicators
        assert indicators["symbol"] == "AAPL"


def test_calculate_technical_indicators_no_data():
    service = MarketDataService()

    # Test with no historical data available
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty DataFrame
        mock_ticker.return_value = mock_ticker_instance

        indicators = service.calculate_technical_indicators("INVALID")

        assert indicators is None


def test_cache_functionality():
    service = MarketDataService()

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {"currentPrice": 150.25, "volume": 1000000}
        mock_ticker.return_value = mock_ticker_instance

        # First call
        data1 = service.get_real_time_data("AAPL")

        # Second call should use cache
        data2 = service.get_real_time_data("AAPL")

        assert data1 == data2
        assert "AAPL" in service.cache
