# tests/test_api.py
"""Smoke tests for the FastAPI inference endpoint."""
import pytest
from fastapi.testclient import TestClient
from app.main import app  # Import the FastAPI app from main.py

client = TestClient(app)

def test_api_docs_accessible():
    """Verifies FastAPI boots and /docs is accessible."""
    try:
        resp = client.get("/docs")
        assert resp.status_code == 200
    except ImportError:
        pytest.skip("app module not found — check import path")

def test_api_health_endpoint():
    """Tests /health or root endpoint returns 200."""
    try:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    except ImportError:
        pytest.skip("app module not found")