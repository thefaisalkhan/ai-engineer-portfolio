"""
pytest tests for the prediction API.
Tests: happy path, validation, error handling, health check.
"""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ─── Health ───────────────────────────────────────────────────────────────────

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# ─── Predict — Happy Path ─────────────────────────────────────────────────────

def test_predict_returns_label(client):
    resp = client.post("/predict", json={"text": "This is a great AI engineering portfolio!"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in ("positive", "negative", "neutral", "uncertain")
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["latency_ms"] >= 0
    assert "request_id" in data


def test_predict_long_text_is_positive(client):
    long_text = "AI engineering " * 30
    resp = client.post("/predict", json={"text": long_text})
    assert resp.status_code == 200
    assert resp.json()["label"] == "positive"


def test_predict_threshold_uncertain(client):
    resp = client.post("/predict", json={"text": "ok", "threshold": 0.99})
    assert resp.status_code == 200
    assert resp.json()["label"] == "uncertain"


# ─── Validation Errors ────────────────────────────────────────────────────────

def test_predict_empty_text_fails(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_predict_missing_text_fails(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_predict_invalid_threshold(client):
    resp = client.post("/predict", json={"text": "hello", "threshold": 1.5})
    assert resp.status_code == 422


def test_predict_text_stripped(client):
    resp = client.post("/predict", json={"text": "  hello world  "})
    assert resp.status_code == 200


# ─── Docs Available ───────────────────────────────────────────────────────────

def test_docs_available(client):
    resp = client.get("/docs")
    assert resp.status_code == 200
