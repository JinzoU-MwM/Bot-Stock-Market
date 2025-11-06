import pytest
from dashboard.app import create_app


def test_dashboard_home():
    app = create_app()
    client = app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    assert b"AI Trading Dashboard" in response.data


def test_api_system_status():
    app = create_app()
    client = app.test_client()

    response = client.get("/api/system_status")
    assert response.status_code == 200
    data = response.get_json()
    assert "agents" in data
    assert "total_pnl" in data
