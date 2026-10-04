"""
Shared configuration and constants for the MLOps platform.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic_settings import BaseSettings

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"
REFERENCE_DIR = PROJECT_ROOT / "reference"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"


# ── Enums ────────────────────────────────────────────────────────────────────
class ModelStage(StrEnum):
    """Explicit model lifecycle states (spec §02)."""

    CANDIDATE = "CANDIDATE"
    VALIDATING = "VALIDATING"
    REGISTERED = "REGISTERED"
    CHAMPION = "CHAMPION"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


class DriftSeverity(StrEnum):
    """Drift monitoring severity levels (spec §07)."""

    OK = "OK"
    WARNING = "WARNING"
    RETRAIN_REQUIRED = "RETRAIN_REQUIRED"
    DEGRADED = "DEGRADED"  # insufficient samples


class RetrainTrigger(StrEnum):
    """Retraining trigger sources (spec §08)."""

    SCHEDULED = "SCHEDULED"
    DRIFT = "DRIFT"
    MANUAL = "MANUAL"
    DATA_REFRESH = "DATA_REFRESH"
    EXPERIMENT = "EXPERIMENT"


class AuditEvent(StrEnum):
    """Audit event types (spec §11)."""

    TRAIN = "TRAIN"
    REGISTER = "REGISTER"
    VALIDATE = "VALIDATE"
    PROMOTE = "PROMOTE"
    ROLLBACK = "ROLLBACK"
    REJECT = "REJECT"
    DRIFT_ALERT = "DRIFT_ALERT"


# ── Settings ─────────────────────────────────────────────────────────────────
class AppSettings(BaseSettings):
    """Application settings loaded from environment."""

    # Service
    service_name: str = "mlops-fraud-detection"
    service_version: str = "1.0.0"
    debug: bool = False

    # Model
    model_name: str = "credit-card-fraud"
    model_dir: str = str(MODELS_DIR)
    default_model_version: str = "1.0.0"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_batch_size: int = 100
    max_request_body_bytes: int = 10 * 1024 * 1024  # 10 MB

    # Drift
    drift_min_samples: int = 200
    drift_warning_feature_fraction: float = 0.20
    drift_retrain_feature_fraction: float = 0.40
    drift_prediction_warning: float = 0.20
    drift_prediction_retrain: float = 0.40

    # Promotion gates
    # For classification: candidate F1 must be >= champion_f1 * (1 - tolerance)
    promotion_metric: str = "f1_score"
    promotion_tolerance: float = 0.02

    # Registry
    registry_backend: str = "local"  # "local" | "mlflow"
    mlflow_tracking_uri: str = "http://localhost:5000"

    model_config = {"env_prefix": "MLOPS_", "env_file": ".env", "extra": "ignore"}


def get_settings() -> AppSettings:
    """Get cached application settings."""
    return AppSettings()
