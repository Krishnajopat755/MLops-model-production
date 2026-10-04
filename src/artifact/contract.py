"""
Artifact contract — manifest and schema models.

Every deployable version must include the model, preprocessing,
feature schema, training config, metrics, dataset reference,
source commit, and dependency hash (spec §04 / FR-02).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FeatureSpec(BaseModel):
    """Single feature definition."""

    name: str
    type: str = "float64"
    required: bool = True


class FeatureSchema(BaseModel):
    """Feature schema contract for the model."""

    features: list[FeatureSpec]
    target: dict[str, str] | None = None
    version: str = "1"


class ModelManifest(BaseModel):
    """
    Immutable manifest for a deployable model version (spec §04).

    This is the contract between training and serving.
    """

    model_name: str
    model_version: str
    training_run_id: str
    git_sha: str
    config_sha256: str
    dataset_fingerprint: str
    feature_schema_version: str = "1"
    algorithm: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    trained_at: str = ""
    artifact_files: list[str] = Field(default_factory=list)


class ArtifactBundle(BaseModel):
    """Complete artifact bundle metadata."""

    manifest: ModelManifest
    feature_schema: FeatureSchema
    artifact_dir: str
    model_path: str
    reference_data_path: str | None = None
