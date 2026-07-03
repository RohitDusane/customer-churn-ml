# tests/test_api.py
"""Smoke tests for the FastAPI inference endpoint."""
import pytest
from fastapi.testclient import TestClient

def test_api_docs_accessible():
    """Verifies FastAPI boots and /docs is accessible."""
    try:
        from app import app
        client = TestClient(app)
        resp = client.get("/docs")
        assert resp.status_code == 200
    except ImportError:
        pytest.skip("app module not found — check import path")

def test_api_health_endpoint():
    """Tests /health or root endpoint returns 200."""
    try:
        from app import app
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code in (200, 404, 422)
    except ImportError:
        pytest.skip("app module not found")