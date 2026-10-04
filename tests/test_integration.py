"""
End-to-end integration tests — spec §12 and ANTIGRAVITY_HANDOFF milestones.

Tests:
  - Milestone 1: config -> training -> artifact -> loader -> FastAPI /predict
  - Milestone 2: candidate validation -> promotion -> rollback lifecycle
  - Rejection gate: degraded candidate rejected before promotion
  - Drift evaluation: shifted distribution detection and report generation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, predictor
from src.artifact.loader import ArtifactLoader
from src.config import AppSettings, AuditEvent, DriftSeverity
from src.monitoring.drift import DriftMonitor
from src.registry.local_registry import LocalRegistry
from src.retraining.pipeline import RetrainingPipeline

pytestmark = pytest.mark.integration


class TestMilestone1Integration:
    """
    Milestone 1 integration:
    config -> trained artifact -> Project 3 loader -> local FastAPI -> /predict
    """

    def test_training_to_serving_pipeline(self, tmp_dir: Path) -> None:
        # 1. Create synthetic credit card dataset
        n_samples = 200
        n_features = 29
        feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        X_data = np.random.randn(n_samples, n_features)
        y_data = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

        df = pd.DataFrame(X_data, columns=feature_names)
        df["Class"] = y_data
        data_csv = tmp_dir / "test_creditcard.csv"
        df.to_csv(data_csv, index=False)

        # 2. Build training config
        config = {
            "model_name": "credit-card-fraud",
            "model_version": "1.0.0",
            "data_path": str(data_csv),
            "target_column": "Class",
            "drop_columns": [],
            "algorithm": "random_forest",
            "model_params": {
                "n_estimators": 5,
                "max_depth": 5,
                "random_state": 42,
            },
            "test_size": 0.25,
            "random_state": 42,
        }
        config_path = tmp_dir / "test_config.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # 3. Train model
        output_dir = tmp_dir / "artifact_v1"
        from src.ml_research.train import train_model

        result = train_model(config_path, output_dir, data_path=data_csv)
        assert Path(result["model_path"]).exists()
        assert (output_dir / "manifest.json").exists()
        assert (output_dir / "feature_schema.json").exists()
        assert (output_dir / "reference_data.parquet").exists()

        # 4. Project 3 Loader verifies contract
        loader = ArtifactLoader(output_dir)
        errors = loader.validate_integrity()
        assert errors == []
        bundle = loader.load_bundle()
        assert bundle.manifest.model_name == "credit-card-fraud"

        # 5. Load into predictor
        predictor.load_from_directory(str(output_dir))
        assert predictor.is_ready

        # 6. FastAPI prediction request
        client = TestClient(app)
        sample_req = {f"V{i}": 0.5 for i in range(1, 29)}
        sample_req["Amount"] = 100.0

        resp = client.post("/api/v1/predict", json={"features": sample_req})
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "credit-card-fraud"
        assert data["model_version"] == "1.0.0"
        assert data["prediction"] in (0, 1)
        assert data["label"] in ("fraud", "legitimate")
        assert "request_id" in data


class TestMilestone2LifecycleIntegration:
    """
    Milestone 2 integration:
    versioned model -> drift report -> retrain candidate -> validate -> promote -> rollback
    """

    def test_full_candidate_lifecycle(self, tmp_dir: Path, mock_model_dir: Path) -> None:
        registry_dir = tmp_dir / "registry"
        registry = LocalRegistry(registry_dir)
        settings = AppSettings(model_dir=str(registry_dir))
        pipeline = RetrainingPipeline(registry=registry, settings=settings)

        # 1. Register and promote version 1.0.0
        registry.register(mock_model_dir, "1.0.0")
        registry.promote("credit-card-fraud", "1.0.0")
        assert registry.get_champion_version("credit-card-fraud") == "1.0.0"

        # 2. Prepare synthetic candidate data
        cand_dir = tmp_dir / "cand_1_1"
        cand_dir.mkdir(parents=True)
        # Copy mock model files to candidate
        import shutil

        shutil.copytree(mock_model_dir, cand_dir, dirs_exist_ok=True)
        # Update manifest to 1.1.0
        with open(cand_dir / "manifest.json") as f:
            cand_manifest = json.load(f)
        cand_manifest["model_version"] = "1.1.0"
        with open(cand_dir / "manifest.json", "w") as f:
            json.dump(cand_manifest, f)

        # 3. Validate candidate gates
        pipeline._validate_candidate(cand_dir, "1.1.0")

        # 4. Compare with champion (equal metrics should pass gate)
        comp = pipeline._compare_with_champion(cand_manifest["metrics"])
        assert comp["passes_gate"] is True

        # 5. Register and promote 1.1.0
        registry.register(cand_dir, "1.1.0")
        registry.promote("credit-card-fraud", "1.1.0")
        assert registry.get_champion_version("credit-card-fraud") == "1.1.0"

        # 6. Rollback to 1.0.0
        prev = registry.rollback("credit-card-fraud")
        assert prev == "1.0.0"
        assert registry.get_champion_version("credit-card-fraud") == "1.0.0"

        # 7. Verify audit history
        history = registry.get_history("credit-card-fraud")
        events = [h["event"] for h in history]
        assert events == [
            AuditEvent.REGISTER.value,
            AuditEvent.PROMOTE.value,
            AuditEvent.REGISTER.value,
            AuditEvent.PROMOTE.value,
            AuditEvent.PROMOTE.value,
            AuditEvent.ROLLBACK.value,
        ]

    def test_bad_candidate_rejected_at_validation(self, tmp_dir: Path) -> None:
        """Verify that a corrupted candidate bundle is rejected and not registered."""
        registry_dir = tmp_dir / "registry"
        registry = LocalRegistry(registry_dir)
        settings = AppSettings(model_dir=str(registry_dir))
        pipeline = RetrainingPipeline(registry=registry, settings=settings)

        # Corrupt candidate: missing feature_schema.json
        bad_cand = tmp_dir / "bad_candidate"
        bad_cand.mkdir(parents=True)
        (bad_cand / "manifest.json").write_text(json.dumps({"model_name": "credit-card-fraud"}))

        from src.retraining.pipeline import CandidateValidationError

        with pytest.raises(CandidateValidationError):
            pipeline._validate_candidate(bad_cand, "9.9.9")

    def test_drift_to_retrain_trigger(self, tmp_dir: Path, mock_model_dir: Path) -> None:
        """Verify drift detection triggers RETRAIN_REQUIRED under severe distribution shift."""
        settings = AppSettings(
            drift_min_samples=50,
            drift_retrain_feature_fraction=0.30,
        )
        monitor = DriftMonitor(settings=settings)
        monitor.load_reference(mock_model_dir / "reference_data.parquet")

        # Create severely drifted distribution
        ref_df = pd.read_parquet(mock_model_dir / "reference_data.parquet")
        feature_cols = [c for c in ref_df.columns if not c.startswith("_")]
        shifted_df = ref_df[feature_cols].copy()
        # Shift all features by 10 standard deviations
        shifted_df = shifted_df + 10.0

        report = monitor.evaluate_drift(shifted_df)
        assert report["severity"] == DriftSeverity.RETRAIN_REQUIRED.value
        assert report["drifted_fraction"] > 0.50

        # Save report
        report_dir = monitor.save_report(report, tmp_dir / "reports")
        assert (report_dir / "summary.json").exists()
        assert (report_dir / "feature_drift.csv").exists()
