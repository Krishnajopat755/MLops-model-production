"""
Model serving — loads and manages the in-memory model.

The predictor loads a validated champion from the registry at startup.
It never accepts a model path from an HTTP request (spec rule #5).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.artifact.contract import ArtifactBundle, FeatureSchema
from src.artifact.loader import ArtifactLoader

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Thread-safe model predictor.

    Loads a champion model at startup. Supports hot-reload
    via load-validate-swap pattern (spec §11).
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._bundle: ArtifactBundle | None = None
        self._lock = threading.Lock()
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def model_name(self) -> str:
        if self._bundle:
            return self._bundle.manifest.model_name
        return "unknown"

    @property
    def model_version(self) -> str:
        if self._bundle:
            return self._bundle.manifest.model_version
        return "unknown"

    @property
    def feature_schema(self) -> FeatureSchema | None:
        if self._bundle:
            return self._bundle.feature_schema
        return None

    @property
    def feature_names(self) -> list[str]:
        if self._bundle:
            return [f.name for f in self._bundle.feature_schema.features]
        return []

    def load_from_directory(self, artifact_dir: str) -> None:
        """
        Load and validate a model from an artifact directory.

        Uses load-validate-swap: the new model is fully loaded and
        validated before replacing the in-memory handle (spec §11).
        """
        loader = ArtifactLoader(artifact_dir)

        # Validate
        errors = loader.validate_integrity()
        if errors:
            raise ValueError(f"Artifact integrity errors: {errors}")

        bundle = loader.load_bundle()
        pipeline = loader.load_model()

        # Smoke test before swap
        if not loader.smoke_test(pipeline, bundle.feature_schema):
            raise ValueError("Model smoke test failed — not swapping")

        # Atomic swap
        with self._lock:
            self._pipeline = pipeline
            self._bundle = bundle
            self._ready = True

        logger.info(
            "Loaded model %s v%s (%d features)",
            bundle.manifest.model_name,
            bundle.manifest.model_version,
            len(bundle.feature_schema.features),
        )

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Run a single prediction.

        Input is validated against the feature schema.
        Returns prediction, probability, and model metadata.
        """
        if not self._ready or self._pipeline is None or self._bundle is None:
            raise RuntimeError("Model not loaded")

        schema = self._bundle.feature_schema
        feature_names = [f.name for f in schema.features]

        # Validate input
        self._validate_input(features, feature_names)

        # Build feature array in correct order
        values = [[features[name] for name in feature_names]]
        X = pd.DataFrame(values, columns=feature_names)

        with self._lock:
            prediction = int(self._pipeline.predict(X)[0])
            probability = None
            if hasattr(self._pipeline, "predict_proba"):
                proba = self._pipeline.predict_proba(X)[0]
                probability = float(proba[1])  # fraud probability

        return {
            "prediction": prediction,
            "fraud_probability": probability,
            "label": "fraud" if prediction == 1 else "legitimate",
        }

    def predict_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run batch predictions."""
        if not self._ready or self._pipeline is None or self._bundle is None:
            raise RuntimeError("Model not loaded")

        schema = self._bundle.feature_schema
        feature_names = [f.name for f in schema.features]

        # Validate all inputs
        for i, features in enumerate(batch):
            try:
                self._validate_input(features, feature_names)
            except ValueError as e:
                raise ValueError(f"Batch item {i}: {e}") from e

        # Build dataframe
        rows = [[item[name] for name in feature_names] for item in batch]
        X = pd.DataFrame(rows, columns=feature_names)

        with self._lock:
            predictions = self._pipeline.predict(X).tolist()
            probabilities = None
            if hasattr(self._pipeline, "predict_proba"):
                probabilities = self._pipeline.predict_proba(X)[:, 1].tolist()

        results = []
        for i, pred in enumerate(predictions):
            result: dict[str, Any] = {
                "prediction": int(pred),
                "label": "fraud" if int(pred) == 1 else "legitimate",
            }
            if probabilities is not None:
                result["fraud_probability"] = float(probabilities[i])
            results.append(result)

        return results

    def _validate_input(self, features: dict[str, Any], expected: list[str]) -> None:
        """Validate input features against schema (spec §05)."""
        # Check missing features
        missing = set(expected) - set(features.keys())
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")

        # Check for NaN/infinity
        for name in expected:
            val = features[name]
            if not isinstance(val, (int, float)):
                raise ValueError(f"Feature '{name}' must be numeric, got {type(val).__name__}")
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"Feature '{name}' must be finite, got {val}")
