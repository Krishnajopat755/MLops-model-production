"""
Credit Card Fraud Detection — training pipeline.

This is the 'existing ml-research' training logic. Project 3 reuses
this module via the adapter; it NEVER duplicates the preprocessing or
model construction code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Data loading ─────────────────────────────────────────────────────────────
def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load the credit card fraud CSV."""
    df = pd.read_csv(data_path)
    logger.info("Loaded dataset: %d rows, %d columns", len(df), len(df.columns))
    return df


# ── Preprocessing ────────────────────────────────────────────────────────────
def build_preprocessing_pipeline(feature_columns: list[str]) -> Pipeline:
    """
    Build the preprocessing pipeline.

    The scaler is fitted during training and packaged with the model
    to prevent training-serving skew (spec §04).
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )


def prepare_features(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split dataframe into features and target."""
    if drop_columns:
        df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors="ignore")

    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target_column]

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Replace infinities with NaN, then fill NaN with 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, y, feature_columns


# ── Model construction ───────────────────────────────────────────────────────
def build_model(config: dict[str, Any]) -> Pipeline:
    """
    Build a full sklearn pipeline: preprocessing + classifier.

    The pipeline packages preprocessing WITH the model so the serving
    artifact preserves training-time transformations (spec §04).
    """
    algorithm = config.get("algorithm", "random_forest")
    model_params = config.get("model_params", {})

    if algorithm == "logistic_regression":
        classifier = LogisticRegression(**model_params)
    elif algorithm == "random_forest":
        classifier = RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


# ── Training ─────────────────────────────────────────────────────────────────
def train_model(
    config_path: str | Path,
    output_dir: str | Path,
    data_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Train the model from a YAML config and save artifacts.

    Returns a dict with artifact paths and metrics.
    """
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve data path
    if data_path is None:
        data_path = config.get("data_path")
    if data_path is None:
        raise ValueError("data_path must be provided in config or as argument")

    # Load and prepare data
    df = load_dataset(data_path)
    target_col = config.get("target_column", "Class")
    drop_cols = config.get("drop_columns", [])
    feature_cols = config.get("feature_columns", None)

    X, y, resolved_features = prepare_features(df, target_col, feature_cols, drop_cols)

    # Train/test split
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Build and train
    pipeline = build_model(config)
    logger.info("Training %s model...", config.get("algorithm", "random_forest"))
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics["classification_report"] = report

    logger.info("Metrics: %s", {k: v for k, v in metrics.items() if k != "classification_report"})

    # ── Save artifacts ───────────────────────────────────────────────────
    # Model
    model_path = output_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    # Metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Feature schema
    schema = {
        "features": [
            {"name": feat, "type": "float64", "required": True} for feat in resolved_features
        ],
        "target": {"name": target_col, "type": "int64"},
        "version": "1",
    }
    schema_path = output_dir / "feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    # Training config copy
    config_copy_path = output_dir / "training_config.yaml"
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Config hash
    with open(config_path, "rb") as f:
        config_sha256 = hashlib.sha256(f.read()).hexdigest()

    # Dataset fingerprint
    dataset_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:16]

    # Evaluation results
    evaluation = {
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "fraud_ratio": float(y.mean()),
        "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
        "evaluated_at": datetime.now(UTC).isoformat(),
    }
    eval_path = output_dir / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2)

    # Reference data for drift monitoring (save training distribution)
    reference_data = X_train.copy()
    reference_data["_prediction"] = pipeline.predict(X_train)
    if hasattr(pipeline, "predict_proba"):
        reference_data["_probability"] = pipeline.predict_proba(X_train)[:, 1]
    ref_path = output_dir / "reference_data.parquet"
    reference_data.to_parquet(ref_path, index=False)

    # Manifest
    run_id = str(uuid.uuid4())
    manifest = {
        "model_name": config.get("model_name", "credit-card-fraud"),
        "model_version": config.get("model_version", "1.0.0"),
        "training_run_id": run_id,
        "git_sha": _get_git_sha(),
        "config_sha256": config_sha256,
        "dataset_fingerprint": dataset_fingerprint,
        "feature_schema_version": "1",
        "algorithm": config.get("algorithm", "random_forest"),
        "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
        "trained_at": datetime.now(UTC).isoformat(),
        "artifact_files": [
            "model.joblib",
            "metrics.json",
            "feature_schema.json",
            "training_config.yaml",
            "evaluation.json",
            "reference_data.parquet",
            "manifest.json",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Artifacts saved to %s", output_dir)

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
        "manifest": manifest,
    }


def _get_git_sha() -> str:
    """Attempt to get the current git SHA."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(
    model_path: str | Path,
    data_path: str | Path,
    target_column: str = "Class",
    feature_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a saved model on a dataset."""
    pipeline = joblib.load(model_path)
    df = load_dataset(data_path)
    X, y, _ = prepare_features(df, target_column, feature_columns, drop_columns)

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, "predict_proba") else None

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y, y_proba))

    return metrics


# ── Model loading ────────────────────────────────────────────────────────────
def load_model(model_path: str | Path) -> Pipeline:
    """Load a serialized model pipeline."""
    return joblib.load(model_path)


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train credit card fraud detection model")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--output", default="models/credit-card-fraud/1.0.0", help="Output dir")
    parser.add_argument("--data", default=None, help="Override data path from config")
    args = parser.parse_args()

    result = train_model(args.config, args.output, args.data)
    print(json.dumps(result["metrics"], indent=2))
