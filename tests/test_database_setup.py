import pytest
from unittest.mock import Mock, patch
from database.connection import DatabaseConnection


@patch("psycopg2.connect")
def test_database_connection(mock_connect):
    # Mock connection
    mock_conn = Mock()
    mock_cursor = Mock()

    # Setup mock to return successful connection
    mock_connect.return_value = mock_conn

    # Mock cursor context manager
    mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)

    # Mock SELECT 1 query result
    mock_cursor.fetchall.return_value = [(1,)]

    db = DatabaseConnection()
    assert db.connect() == True
    assert db.execute("SELECT 1") == [(1,)]


@patch("psycopg2.connect")
def test_tables_created(mock_connect):
    # Mock connection
    mock_conn = Mock()
    mock_cursor = Mock()

    # Setup mock
    mock_connect.return_value = mock_conn

    # Mock cursor context manager
    mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)

    # Mock table names query result
    mock_cursor.fetchall.return_value = [
        {"table_name": "agents"},
        {"table_name": "trades"},
        {"table_name": "market_data"},
        {"table_name": "experiences"},
        {"table_name": "models"},
        {"table_name": "performance_summary"},
    ]

    db = DatabaseConnection()
    db.connect()
    tables = db.get_table_names()
    expected_tables = [
        "agents",
        "trades",
        "market_data",
        "experiences",
        "models",
        "performance_summary",
    ]
    for table in expected_tables:
        assert table in tables
