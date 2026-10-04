"""
Unit tests — artifact contract and loader (spec §12).

Tests:
  - manifest validation
  - feature schema validation
  - artifact integrity
  - loader error handling
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.artifact.contract import FeatureSchema, FeatureSpec, ModelManifest
from src.artifact.loader import ArtifactLoader

pytestmark = pytest.mark.unit


class TestModelManifest:
    """Test manifest validation."""

    def test_valid_manifest(self, sample_manifest: dict[str, Any]) -> None:
        manifest = ModelManifest(**sample_manifest)
        assert manifest.model_name == "credit-card-fraud"
        assert manifest.model_version == "1.0.0"
        assert manifest.metrics["f1_score"] == 0.85

    def test_manifest_requires_model_name(self) -> None:
        with pytest.raises(ValidationError):
            ModelManifest(model_version="1.0.0")  # type: ignore


class TestFeatureSchema:
    """Test feature schema validation."""

    def test_valid_schema(self, sample_feature_schema: dict[str, Any]) -> None:
        schema = FeatureSchema(**sample_feature_schema)
        assert len(schema.features) == 29
        assert schema.version == "1"

    def test_feature_spec(self) -> None:
        spec = FeatureSpec(name="Amount", type="float64", required=True)
        assert spec.name == "Amount"
        assert spec.required is True


class TestArtifactLoader:
    """Test artifact loader."""

    def test_load_bundle(self, mock_model_dir: Path) -> None:
        loader = ArtifactLoader(mock_model_dir)
        bundle = loader.load_bundle()
        assert bundle.manifest.model_name == "credit-card-fraud"
        assert len(bundle.feature_schema.features) == 29

    def test_integrity_check_passes(self, mock_model_dir: Path) -> None:
        loader = ArtifactLoader(mock_model_dir)
        errors = loader.validate_integrity()
        assert errors == []

    def test_integrity_check_fails_missing_file(self, tmp_dir: Path) -> None:
        # Create partial artifact
        (tmp_dir / "manifest.json").write_text("{}")
        loader = ArtifactLoader(tmp_dir)
        errors = loader.validate_integrity()
        assert len(errors) > 0

    def test_smoke_test_passes(self, mock_model_dir: Path) -> None:
        loader = ArtifactLoader(mock_model_dir)
        bundle = loader.load_bundle()
        pipeline = loader.load_model()
        assert loader.smoke_test(pipeline, bundle.feature_schema) is True

    def test_load_nonexistent_dir(self) -> None:
        with pytest.raises(FileNotFoundError):
            ArtifactLoader("/nonexistent/path")
