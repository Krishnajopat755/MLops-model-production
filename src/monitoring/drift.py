"""
Drift monitoring — spec §07.

Monitors feature distributions (data drift) and prediction distributions
(prediction drift) against immutable, model-version-specific references.

Drift is a monitoring signal — it is NOT, by itself, proof that predictive
performance has degraded (ADR-006).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config import AppSettings, DriftSeverity, get_settings

logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Drift detection engine.

    Two drift types (spec §07):
      1. Data drift: feature distribution changes vs training reference
      2. Prediction drift: output distribution changes vs reference predictions
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.reference_data: pd.DataFrame | None = None
        self.reference_predictions: pd.Series | None = None

    def load_reference(self, reference_path: str | Path) -> None:
        """Load immutable reference data for drift comparison."""
        self.reference_data = pd.read_parquet(reference_path)

        if "_prediction" in self.reference_data.columns:
            self.reference_predictions = self.reference_data["_prediction"]
            self.reference_data = self.reference_data.drop(
                columns=["_prediction", "_probability"], errors="ignore"
            )

        logger.info(
            "Loaded reference data: %d rows, %d features",
            len(self.reference_data),
            len(self.reference_data.columns),
        )

    def calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Population Stability Index (PSI).

        PSI measures the shift between two distributions.
        """
        eps = 1e-6

        # Create bins from reference
        breakpoints = np.linspace(
            min(reference.min(), current.min()) - eps,
            max(reference.max(), current.max()) + eps,
            bins + 1,
        )

        ref_counts = np.histogram(reference, bins=breakpoints)[0] + eps
        cur_counts = np.histogram(current, bins=breakpoints)[0] + eps

        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return psi

    def calculate_ks(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Kolmogorov-Smirnov statistic."""
        statistic, _ = stats.ks_2samp(reference, current)
        return float(statistic)

    def check_feature_drift(self, current_data: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """
        Check data drift for each feature.

        Returns per-feature PSI, KS, and drift status.
        """
        if self.reference_data is None:
            raise ValueError("Reference data not loaded")

        results: dict[str, dict[str, Any]] = {}
        common_cols = [c for c in self.reference_data.columns if c in current_data.columns]

        for col in common_cols:
            ref = self.reference_data[col].dropna().values.astype(float)
            cur = current_data[col].dropna().values.astype(float)

            if len(ref) == 0 or len(cur) == 0:
                results[col] = {"psi": 0.0, "ks": 0.0, "drifted": False}
                continue

            psi = self.calculate_psi(ref, cur)
            ks = self.calculate_ks(ref, cur)

            # Feature is drifted if PSI > 0.2 (moderate shift)
            drifted = psi > 0.2

            results[col] = {
                "psi": round(psi, 4),
                "ks": round(ks, 4),
                "drifted": drifted,
            }

        return results

    def check_prediction_drift(self, current_predictions: np.ndarray | pd.Series) -> dict[str, Any]:
        """Check prediction distribution drift."""
        if self.reference_predictions is None:
            raise ValueError("Reference predictions not loaded")

        ref = self.reference_predictions.values.astype(float)
        cur = np.asarray(current_predictions, dtype=float)

        psi = self.calculate_psi(ref, cur)
        ks = self.calculate_ks(ref, cur)

        return {
            "psi": round(psi, 4),
            "ks": round(ks, 4),
            "drifted": psi > 0.2,
        }

    def evaluate_drift(
        self,
        current_data: pd.DataFrame,
        current_predictions: np.ndarray | pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Full drift evaluation with composite policy (spec §07).

        Returns severity: OK, WARNING, RETRAIN_REQUIRED, or DEGRADED.
        """
        n_samples = len(current_data)

        # Minimum sample gate (spec §07)
        if n_samples < self.settings.drift_min_samples:
            return {
                "severity": DriftSeverity.DEGRADED.value,
                "reason": f"Insufficient samples: {n_samples} < {self.settings.drift_min_samples}",
                "n_samples": n_samples,
                "feature_drift": {},
                "prediction_drift": None,
                "evaluated_at": datetime.now(UTC).isoformat(),
            }

        # Feature drift
        feature_drift = self.check_feature_drift(current_data)
        total_features = len(feature_drift)
        drifted_features = sum(1 for f in feature_drift.values() if f["drifted"])
        drifted_fraction = drifted_features / max(total_features, 1)

        # Prediction drift
        prediction_drift = None
        if current_predictions is not None and self.reference_predictions is not None:
            prediction_drift = self.check_prediction_drift(current_predictions)

        # Composite policy (spec §07)
        severity = DriftSeverity.OK

        # Check feature drift thresholds
        if drifted_fraction >= self.settings.drift_retrain_feature_fraction:
            severity = DriftSeverity.RETRAIN_REQUIRED
        elif drifted_fraction >= self.settings.drift_warning_feature_fraction:
            severity = DriftSeverity.WARNING

        # Check prediction drift thresholds
        if prediction_drift and prediction_drift["psi"] >= self.settings.drift_prediction_retrain:
            severity = DriftSeverity.RETRAIN_REQUIRED
        elif (
            prediction_drift
            and prediction_drift["psi"] >= self.settings.drift_prediction_warning
            and severity == DriftSeverity.OK
        ):
            severity = DriftSeverity.WARNING

        return {
            "severity": severity.value,
            "n_samples": n_samples,
            "total_features": total_features,
            "drifted_features": drifted_features,
            "drifted_fraction": round(drifted_fraction, 4),
            "feature_drift": feature_drift,
            "prediction_drift": prediction_drift,
            "evaluated_at": datetime.now(UTC).isoformat(),
        }

    def save_report(
        self,
        report: dict[str, Any],
        output_dir: str | Path,
        run_id: str | None = None,
    ) -> Path:
        """
        Save drift report (spec §07).

        Output: reports/drift/{run_id}/summary.json
        """
        output_dir = Path(output_dir)
        if run_id:
            output_dir = output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Feature drift CSV
        if report.get("feature_drift"):
            df = pd.DataFrame.from_dict(report["feature_drift"], orient="index")
            df.to_csv(output_dir / "feature_drift.csv")

        # Prediction drift
        if report.get("prediction_drift"):
            with open(output_dir / "prediction_drift.json", "w") as f:
                json.dump(report["prediction_drift"], f, indent=2)

        logger.info("Drift report saved to %s", output_dir)
        return output_dir


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import uuid

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate data and prediction drift")
    parser.add_argument(
        "--reference",
        default="models/credit-card-fraud/1.0.0/reference_data.parquet",
        help="Path to reference parquet file",
    )
    parser.add_argument(
        "--current",
        default="data/creditcard.csv",
        help="Path to current data CSV or Parquet",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of samples to draw from current data",
    )
    parser.add_argument(
        "--output",
        default="reports/drift",
        help="Output directory for drift reports",
    )
    args = parser.parse_args()

    app_settings = get_settings()
    monitor = DriftMonitor(settings=app_settings)
    monitor.load_reference(args.reference)

    # Load current data
    if args.current.endswith(".parquet"):
        curr_df = pd.read_parquet(args.current)
    else:
        curr_df = pd.read_csv(args.current)

    if "Class" in curr_df.columns:
        curr_df = curr_df.drop(columns=["Class"])
    if "Time" in curr_df.columns:
        curr_df = curr_df.drop(columns=["Time"])

    if len(curr_df) > args.sample_size:
        curr_df = curr_df.sample(n=args.sample_size, random_state=42)

    report = monitor.evaluate_drift(curr_df)
    run_id = f"eval_{uuid.uuid4().hex[:8]}"
    saved_dir = monitor.save_report(report, args.output, run_id=run_id)

    print(
        json.dumps(
            {
                "severity": report["severity"],
                "n_samples": report["n_samples"],
                "drifted_features": report["drifted_features"],
                "drifted_fraction": report["drifted_fraction"],
                "report_directory": str(saved_dir),
            },
            indent=2,
        )
    )
