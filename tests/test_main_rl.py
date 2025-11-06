# tests/test_main_rl.py
import pytest
from unittest.mock import patch, MagicMock
from main_rl import create_app, initialize_system


@patch("main_rl.DatabaseConnection")
@patch("main_rl.Coordinator")
def test_system_initialization(mock_coordinator, mock_db):
    # Setup mocks
    mock_db_instance = MagicMock()
    mock_db_instance.connect.return_value = True
    mock_db_instance.get_table_names.return_value = [
        "agents",
        "trades",
        "market_data",
        "experiences",
        "models",
        "performance_summary",
    ]
    mock_db.return_value = mock_db_instance

    mock_coordinator_instance = MagicMock()
    mock_coordinator_instance.agents = [MagicMock() for _ in range(5)]  # 5 agents
    mock_coordinator.return_value = mock_coordinator_instance

    app = create_app()
    with app.app_context():
        coordinator = initialize_system()
        assert coordinator is not None
        assert len(coordinator.agents) > 0


def test_health_check():
    app = create_app()
    client = app.test_client()

    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
