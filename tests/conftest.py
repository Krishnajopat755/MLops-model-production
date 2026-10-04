"""
Shared test fixtures and helpers.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_feature_schema() -> dict[str, Any]:
    """Sample feature schema for credit card fraud."""
    return {
        "features": [{"name": f"V{i}", "type": "float64", "required": True} for i in range(1, 29)]
        + [
            {"name": "Amount", "type": "float64", "required": True},
        ],
        "target": {"name": "Class", "type": "int64"},
        "version": "1",
    }


@pytest.fixture
def sample_manifest() -> dict[str, Any]:
    """Sample model manifest."""
    return {
        "model_name": "credit-card-fraud",
        "model_version": "1.0.0",
        "training_run_id": "test-run-001",
        "git_sha": "abc1234",
        "config_sha256": "deadbeef",
        "dataset_fingerprint": "cafebabe",
        "feature_schema_version": "1",
        "algorithm": "random_forest",
        "metrics": {
            "accuracy": 0.99,
            "precision": 0.90,
            "recall": 0.80,
            "f1_score": 0.85,
            "roc_auc": 0.95,
        },
        "trained_at": "2024-01-01T00:00:00Z",
        "artifact_files": [
            "model.joblib",
            "metrics.json",
            "feature_schema.json",
            "manifest.json",
        ],
    }


@pytest.fixture
def mock_model_dir(
    tmp_dir: Path,
    sample_manifest: dict[str, Any],
    sample_feature_schema: dict[str, Any],
) -> Path:
    """Create a mock model artifact directory with all required files."""
    # Build a simple pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )

    # Train on synthetic data (29 features matching schema)
    feature_names = [f["name"] for f in sample_feature_schema["features"]]
    X = pd.DataFrame(np.random.randn(100, len(feature_names)), columns=feature_names)
    y = np.random.randint(0, 2, 100)
    pipeline.fit(X, y)

    model_dir = tmp_dir / "mock_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    model_path = model_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    manifest_path = model_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(sample_manifest, f, indent=2)

    schema_path = model_dir / "feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(sample_feature_schema, f, indent=2)

    metrics_path = model_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(sample_manifest["metrics"], f, indent=2)

    # Reference data for drift monitoring
    ref_df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)] + ["Amount"])
    ref_df["_prediction"] = pipeline.predict(X)
    ref_df["_probability"] = pipeline.predict_proba(X)[:, 1]
    ref_df.to_parquet(model_dir / "reference_data.parquet", index=False)

    return model_dir


@pytest.fixture
def sample_features() -> dict[str, float]:
    """Sample feature dict for prediction requests."""
    features = {f"V{i}": float(np.random.randn()) for i in range(1, 29)}
    features["Amount"] = 149.62
    return features
