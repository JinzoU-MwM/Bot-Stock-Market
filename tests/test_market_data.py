import pytest
from unittest.mock import Mock, patch
import pandas as pd
from services.market_data import MarketDataService


def test_get_real_time_data():
    """Test basic real-time data fetching - Note: This test requires mocking for CI/CD"""
    service = MarketDataService()

    # Use mocking to avoid API rate limits in automated testing
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

        assert "price" in data
        assert "volume" in data
        assert "timestamp" in data
        assert data["price"] > 0


def test_get_real_time_data_with_mock():
    """Test real-time data fetching with mocking to avoid API calls"""
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
    """Test basic technical indicators calculation"""
    service = MarketDataService()

    # Mock the data to avoid API calls
    with patch.object(service, "get_historical_data") as mock_hist:
        # Create mock historical data with sufficient data points for indicators
        mock_hist_data = pd.DataFrame(
            {
                "Open": [145 + i for i in range(50)],
                "High": [146 + i for i in range(50)],
                "Low": [144 + i for i in range(50)],
                "Close": [145.5 + i * 0.5 for i in range(50)],
                "Volume": [1000000] * 50,
            }
        )
        mock_hist.return_value = mock_hist_data

        indicators = service.calculate_technical_indicators("AAPL")

        assert "rsi" in indicators
        assert "macd" in indicators
        assert "bollinger_upper" in indicators
        assert "bollinger_lower" in indicators


def test_calculate_technical_indicators_with_mock():
    """Test technical indicators calculation with mocking"""
    service = MarketDataService()

    # Mock yfinance historical data
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker_instance = Mock()

        # Create mock historical data with sufficient data points for indicators
        mock_hist = pd.DataFrame(
            {
                "Open": [145 + i for i in range(50)],
                "High": [146 + i for i in range(50)],
                "Low": [144 + i for i in range(50)],
                "Close": [145.5 + i * 0.5 for i in range(50)],
                "Volume": [1000000] * 50,
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
        assert "bollinger_middle" in indicators
        assert "atr" in indicators
        assert "symbol" in indicators
        assert indicators["symbol"] == "AAPL"
        assert indicators["rsi"] is not None
        assert indicators["macd"] is not None
        assert indicators["atr"] is not None


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
    """Test basic caching mechanism"""
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


def test_cache_duration_configuration():
    """Test configurable cache duration"""
    service = MarketDataService()
    service.cache_duration = 30  # Set to 30 seconds

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {"currentPrice": 150.25, "volume": 1000000}
        mock_ticker.return_value = mock_ticker_instance

        # First call
        data1 = service.get_real_time_data("AAPL")

        # Check that cache is populated
        assert "AAPL" in service.cache
        cached_data, timestamp = service.cache["AAPL"]
        assert cached_data == data1


def test_cache_expiry():
    """Test that cache expires after duration"""
    service = MarketDataService()
    service.cache_duration = 0  # Set to 0 for immediate expiry

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {"currentPrice": 150.25, "volume": 1000000}
        mock_ticker.return_value = mock_ticker_instance

        # First call
        data1 = service.get_real_time_data("AAPL")

        # Modify the mock to return different price
        mock_ticker_instance.info = {"currentPrice": 155.50, "volume": 1000000}

        # Second call should fetch new data due to cache expiry
        data2 = service.get_real_time_data("AAPL")

        # Should get new price since cache expired
        assert data2["price"] == 155.50


def test_store_market_data():
    """Test database storage functionality"""
    service = MarketDataService()

    # Mock database connection
    with patch("database.connection.DatabaseConnection") as mock_db:
        mock_db_instance = Mock()
        mock_db.return_value = mock_db_instance

        test_data = {
            "symbol": "AAPL",
            "timestamp": pd.Timestamp.now(),
            "price": 150.25,
            "open": 149.50,
            "high": 151.00,
            "low": 148.75,
            "volume": 1000000,
            "rsi": 65.5,
            "macd": 2.3,
            "bollinger_upper": 155.0,
            "bollinger_lower": 145.0,
        }

        service.store_market_data("AAPL", test_data)

        # Verify database execute was called
        mock_db_instance.execute.assert_called_once()


def test_database_storage_error_handling():
    """Test error handling in database storage"""
    service = MarketDataService()

    # Mock database connection to raise exception
    with patch("database.connection.DatabaseConnection") as mock_db:
        mock_db_instance = Mock()
        mock_db_instance.execute.side_effect = Exception("Database error")
        mock_db.return_value = mock_db_instance

        test_data = {
            "symbol": "AAPL",
            "timestamp": pd.Timestamp.now(),
            "price": 150.25,
            "open": 149.50,
            "high": 151.00,
            "low": 148.75,
            "volume": 1000000,
        }

        # Should not raise exception, should handle gracefully
        service.store_market_data("AAPL", test_data)

        # Still should have attempted to execute
        mock_db_instance.execute.assert_called_once()


def test_atr_calculation():
    """Test ATR (Average True Range) calculation"""
    service = MarketDataService()

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker_instance = Mock()

        # Create mock historical data with realistic price movements
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        prices = [150 + i * 0.5 + (i % 3) * 0.2 for i in range(50)]

        mock_hist = pd.DataFrame(
            {
                "Open": [p - 0.5 for p in prices],
                "High": [p + 0.8 for p in prices],
                "Low": [p - 1.2 for p in prices],
                "Close": prices,
                "Volume": [1000000] * 50,
            },
            index=dates,
        )

        mock_ticker_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_ticker_instance

        indicators = service.calculate_technical_indicators("AAPL")

        assert indicators is not None
        assert "atr" in indicators
        assert indicators["atr"] is not None
        assert indicators["atr"] > 0  # ATR should be positive
