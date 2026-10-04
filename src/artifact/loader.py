"""
Artifact loader — loads model bundles from disk.

The loader validates the manifest and schema before returning
a deployable model handle. It never accepts an arbitrary path
from an HTTP request (spec rule #5).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .contract import ArtifactBundle, FeatureSchema, ModelManifest

logger = logging.getLogger(__name__)


class ArtifactLoader:
    """Load and validate model artifacts from a directory."""

    REQUIRED_FILES = ["model.joblib", "manifest.json", "feature_schema.json"]

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        if not self.artifact_dir.is_dir():
            raise FileNotFoundError(f"Artifact directory not found: {self.artifact_dir}")

    def validate_integrity(self) -> list[str]:
        """Check all required files exist."""
        errors: list[str] = []
        for fname in self.REQUIRED_FILES:
            fpath = self.artifact_dir / fname
            if not fpath.exists():
                errors.append(f"Missing required file: {fname}")
        return errors

    def load_manifest(self) -> ModelManifest:
        """Load and parse the model manifest."""
        manifest_path = self.artifact_dir / "manifest.json"
        with open(manifest_path) as f:
            data = json.load(f)
        return ModelManifest(**data)

    def load_feature_schema(self) -> FeatureSchema:
        """Load and parse the feature schema."""
        schema_path = self.artifact_dir / "feature_schema.json"
        with open(schema_path) as f:
            data = json.load(f)
        return FeatureSchema(**data)

    def load_model(self) -> Pipeline:
        """Load the serialized sklearn pipeline."""
        model_path = self.artifact_dir / "model.joblib"
        pipeline = joblib.load(model_path)
        logger.info("Loaded model from %s", model_path)
        return pipeline

    def smoke_test(self, pipeline: Pipeline, feature_schema: FeatureSchema) -> bool:
        """
        Run a basic smoke test: create dummy input and verify
        the model returns finite predictions.
        """
        try:
            feature_names = [f.name for f in feature_schema.features]
            dummy_input = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
            pred = pipeline.predict(dummy_input)
            if not np.all(np.isfinite(pred)):
                logger.error("Smoke test failed: non-finite prediction")
                return False
            logger.info("Smoke test passed: prediction=%s", pred)
            return True
        except Exception as e:
            logger.error("Smoke test failed: %s", e)
            return False

    def load_bundle(self) -> ArtifactBundle:
        """
        Load the complete artifact bundle with validation.

        Raises ValueError if integrity checks fail.
        """
        # Integrity check
        errors = self.validate_integrity()
        if errors:
            raise ValueError(f"Artifact integrity errors: {errors}")

        manifest = self.load_manifest()
        schema = self.load_feature_schema()

        model_path = str(self.artifact_dir / "model.joblib")
        ref_path = self.artifact_dir / "reference_data.parquet"

        return ArtifactBundle(
            manifest=manifest,
            feature_schema=schema,
            artifact_dir=str(self.artifact_dir),
            model_path=model_path,
            reference_data_path=str(ref_path) if ref_path.exists() else None,
        )
