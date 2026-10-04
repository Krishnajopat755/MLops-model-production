"""
Pydantic request/response models for the API — spec §05.

Every prediction response contains request_id, service_version,
model_name, model_version, and prediction timestamp (FR-04).
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Requests ─────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    """
    Single prediction request.

    Feature fields are validated dynamically against the model's
    feature schema at runtime, so we accept a flexible dict here.
    """

    features: dict[str, float] = Field(
        ...,
        description=(
            "Feature name -> value mapping. All features from the model schema are required."
        ),
        json_schema_extra={"example": {"V1": -1.36, "V2": -0.07, "V3": 2.54, "Amount": 149.62}},
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: dict[str, float]) -> dict[str, float]:
        for key, val in v.items():
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"Feature '{key}' must be a finite number, got {val}")
        return v

    model_config = {"extra": "forbid"}


class BatchPredictionRequest(BaseModel):
    """Batch prediction request (spec §05)."""

    instances: list[dict[str, float]] = Field(
        ...,
        description="List of feature dictionaries.",
        min_length=1,
    )

    @field_validator("instances")
    @classmethod
    def validate_instances(cls, v: list[dict[str, float]]) -> list[dict[str, float]]:
        for idx, inst in enumerate(v):
            for key, val in inst.items():
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    raise ValueError(
                        f"Instance {idx} feature '{key}' must be a finite number, got {val}"
                    )
        return v

    model_config = {"extra": "forbid"}


# ── Responses ────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    """
    Single prediction response (FR-04).

    Always includes request_id, service_version, model metadata,
    and prediction timestamp.
    """

    prediction: int
    fraud_probability: float | None = None
    label: str
    model_name: str
    model_version: str
    model_alias: str = "champion"
    request_id: str
    service_version: str
    served_at: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[dict[str, Any]]
    count: int
    model_name: str
    model_version: str
    model_alias: str = "champion"
    request_id: str
    service_version: str
    served_at: str


class ErrorResponse(BaseModel):
    """
    Error response — never exposes stack traces (spec §05).
    """

    error: dict[str, str]
    request_id: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Readiness response — includes model status."""

    status: str
    service: str
    version: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


class VersionResponse(BaseModel):
    """Version endpoint response."""

    service_name: str
    service_version: str
    model_name: str | None = None
    model_version: str | None = None
    feature_schema_version: str | None = None
