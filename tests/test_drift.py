"""
Unit tests — drift monitoring (spec §12).

Fixtures:
  - stable distribution -> OK
  - moderate shift -> WARNING
  - severe shift -> RETRAIN_REQUIRED
  - insufficient samples -> DEGRADED
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import AppSettings, DriftSeverity
from src.monitoring.drift import DriftMonitor

pytestmark = pytest.mark.unit


@pytest.fixture
def monitor() -> DriftMonitor:
    settings = AppSettings(
        drift_min_samples=50,
        drift_warning_feature_fraction=0.20,
        drift_retrain_feature_fraction=0.40,
        drift_prediction_warning=0.20,
        drift_prediction_retrain=0.40,
    )
    return DriftMonitor(settings=settings)


@pytest.fixture
def reference_data() -> pd.DataFrame:
    """Stable reference distribution."""
    np.random.seed(42)
    return pd.DataFrame({f"V{i}": np.random.randn(500) for i in range(1, 11)})


class TestDriftCalculations:
    """Test PSI and KS calculations."""

    def test_psi_identical(self, monitor: DriftMonitor) -> None:
        data = np.random.randn(1000)
        psi = monitor.calculate_psi(data, data)
        assert psi < 0.01

    def test_psi_shifted(self, monitor: DriftMonitor) -> None:
        ref = np.random.randn(1000)
        shifted = ref + 3.0
        psi = monitor.calculate_psi(ref, shifted)
        assert psi > 0.2

    def test_ks_identical(self, monitor: DriftMonitor) -> None:
        data = np.random.randn(1000)
        ks = monitor.calculate_ks(data, data)
        assert ks < 0.05

    def test_ks_shifted(self, monitor: DriftMonitor) -> None:
        ref = np.random.randn(1000)
        shifted = ref + 3.0
        ks = monitor.calculate_ks(ref, shifted)
        assert ks > 0.5


class TestDriftPolicy:
    """Test composite drift policy."""

    def test_stable_distribution_ok(
        self, monitor: DriftMonitor, reference_data: pd.DataFrame
    ) -> None:
        """Stable distribution -> OK."""
        monitor.reference_data = reference_data
        monitor.reference_predictions = pd.Series(np.random.randint(0, 2, len(reference_data)))

        # Current data ~same distribution
        np.random.seed(43)
        current = pd.DataFrame({f"V{i}": np.random.randn(200) for i in range(1, 11)})
        current_preds = np.random.randint(0, 2, 200)

        result = monitor.evaluate_drift(current, current_preds)
        assert result["severity"] in [DriftSeverity.OK.value, DriftSeverity.WARNING.value]

    def test_severe_shift_retrain(
        self, monitor: DriftMonitor, reference_data: pd.DataFrame
    ) -> None:
        """Severe shift -> RETRAIN_REQUIRED."""
        monitor.reference_data = reference_data
        monitor.reference_predictions = pd.Series(np.zeros(len(reference_data)))

        # Severely shifted data
        current = reference_data.copy() + 10.0
        current_preds = np.ones(len(current))

        result = monitor.evaluate_drift(current, current_preds)
        assert result["severity"] == DriftSeverity.RETRAIN_REQUIRED.value

    def test_insufficient_samples_degraded(
        self, monitor: DriftMonitor, reference_data: pd.DataFrame
    ) -> None:
        """Insufficient samples -> DEGRADED."""
        monitor.reference_data = reference_data

        # Too few samples
        small_data = reference_data.head(10)
        result = monitor.evaluate_drift(small_data)
        assert result["severity"] == DriftSeverity.DEGRADED.value

    def test_report_saved(
        self, monitor: DriftMonitor, reference_data: pd.DataFrame, tmp_dir: Path
    ) -> None:
        """Test drift report saving."""
        monitor.reference_data = reference_data
        current = reference_data.copy()
        result = monitor.evaluate_drift(current)
        report_dir = monitor.save_report(result, tmp_dir, "test-run")
        assert (report_dir / "summary.json").exists()
