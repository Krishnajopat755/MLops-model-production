"""
Security tests (spec §12).

Tests:
  - malicious model path
  - oversized request
  - no stack traces in errors
  - extra fields rejected
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import app, predictor

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def load_mock_model(mock_model_dir: Path) -> None:
    predictor.load_from_directory(str(mock_model_dir))


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestSecurityEndpoints:
    """API security tests."""

    def test_extra_fields_rejected(self, client: TestClient, sample_features: dict) -> None:
        """Extra/unexpected fields should be rejected (spec §05)."""
        r = client.post(
            "/api/v1/predict",
            json={
                "features": sample_features,
                "model_path": "/etc/passwd",  # extra field
            },
        )
        assert r.status_code == 422

    def test_no_stack_traces(self, client: TestClient) -> None:
        """Error responses should not contain stack traces."""
        r = client.post("/api/v1/predict", json={"features": {}})
        assert r.status_code == 422
        data = r.json()
        assert "traceback" not in str(data).lower()
        assert "Traceback" not in str(data)

    def test_wrong_type_feature(self, client: TestClient, sample_features: dict) -> None:
        """String feature values should be rejected."""
        sample_features["V1"] = "malicious_string"
        r = client.post("/api/v1/predict", json={"features": sample_features})
        assert r.status_code == 422

    def test_no_arbitrary_model_path_endpoint(self, client: TestClient) -> None:
        """There should be no endpoint that accepts a model path."""
        # Verify common dangerous endpoints don't exist
        assert client.post("/api/v1/load_model", json={"path": "/tmp/model"}).status_code == 404
        assert client.post("/api/v1/execute", json={"code": "print(1)"}).status_code == 404
        assert client.post("/api/v1/shell", json={"cmd": "ls"}).status_code == 404
