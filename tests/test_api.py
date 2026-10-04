"""
Unit tests — API endpoints (spec §12).

Tests:
  - valid prediction
  - missing feature
  - wrong type
  - oversized batch
  - health/readiness
  - model version metadata
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import app, predictor, settings

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def load_mock_model(mock_model_dir: Path) -> None:
    """Load the mock model before each test."""
    predictor.load_from_directory(str(mock_model_dir))


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoints:
    """Test health probes."""

    def test_liveness(self, client: TestClient) -> None:
        r = client.get("/health/live")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "alive"

    def test_readiness_with_model(self, client: TestClient) -> None:
        r = client.get("/health/ready")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True

    def test_version(self, client: TestClient) -> None:
        r = client.get("/version")
        assert r.status_code == 200
        data = r.json()
        assert data["service_name"] == settings.service_name
        assert data["model_name"] == "credit-card-fraud"


class TestPredictionEndpoint:
    """Test POST /api/v1/predict."""

    def test_valid_prediction(self, client: TestClient, sample_features: dict[str, float]) -> None:
        r = client.post("/api/v1/predict", json={"features": sample_features})
        assert r.status_code == 200
        data = r.json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1]
        assert data["label"] in ["fraud", "legitimate"]
        assert data["model_name"] == "credit-card-fraud"
        assert "request_id" in data
        assert "served_at" in data

    def test_missing_feature(self, client: TestClient) -> None:
        r = client.post("/api/v1/predict", json={"features": {"V1": 1.0}})
        assert r.status_code == 422

    def test_nan_feature(self, client: TestClient, sample_features: dict[str, float]) -> None:
        import json

        payload = json.dumps({"features": {**sample_features, "V1": None}}).replace("null", "NaN")
        r = client.post(
            "/api/v1/predict",
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422

    def test_infinity_feature(self, client: TestClient, sample_features: dict[str, float]) -> None:
        import json

        payload = json.dumps({"features": {**sample_features, "V1": None}}).replace(
            "null", "Infinity"
        )
        r = client.post(
            "/api/v1/predict",
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422

    def test_request_id_in_response(
        self, client: TestClient, sample_features: dict[str, float]
    ) -> None:
        r = client.post("/api/v1/predict", json={"features": sample_features})
        assert "X-Request-ID" in r.headers
        data = r.json()
        assert data["request_id"] == r.headers["X-Request-ID"]


class TestBatchPrediction:
    """Test POST /api/v1/predict/batch."""

    def test_valid_batch(self, client: TestClient, sample_features: dict[str, float]) -> None:
        r = client.post(
            "/api/v1/predict/batch",
            json={"instances": [sample_features, sample_features]},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_oversized_batch(self, client: TestClient, sample_features: dict[str, float]) -> None:
        batch = [sample_features] * (settings.max_batch_size + 1)
        r = client.post("/api/v1/predict/batch", json={"instances": batch})
        assert r.status_code == 422

    def test_empty_batch(self, client: TestClient) -> None:
        r = client.post("/api/v1/predict/batch", json={"instances": []})
        assert r.status_code == 422
