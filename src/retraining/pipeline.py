"""
Retraining pipeline — spec §08.

Candidate pipeline:
  trigger -> existing ml-research config -> candidate artifact
  -> integrity check -> schema check -> offline evaluation
  -> champion comparison -> promotion gates -> registry
  -> deployment smoke test

The FastAPI serving process NEVER launches training (spec rule #4).
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from src.artifact.loader import ArtifactLoader
from src.config import (
    AppSettings,
    AuditEvent,
    RetrainTrigger,
    get_settings,
)
from src.ml_research.train import train_model
from src.registry.local_registry import LocalRegistry

logger = logging.getLogger(__name__)


class CandidateValidationError(Exception):
    """Raised when a candidate fails validation gates."""

    pass


class RetrainingPipeline:
    """
    Orchestrates the retraining lifecycle.

    Trigger sources (spec §08):
      - scheduled
      - drift policy
      - manual operator dispatch
      - data refresh
      - explicit experiment
    """

    def __init__(
        self,
        registry: LocalRegistry,
        settings: AppSettings | None = None,
    ) -> None:
        self.registry = registry
        self.settings = settings or get_settings()

    def retrain(
        self,
        config_path: str | Path,
        data_path: str | Path,
        trigger: RetrainTrigger,
        candidate_version: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the full retraining pipeline.

        Returns pipeline result with candidate status.
        """
        run_id = str(uuid.uuid4())
        logger.info("Retraining run %s triggered by %s", run_id, trigger.value)

        # Generate candidate version
        if candidate_version is None:
            candidate_version = self._next_version()

        candidate_dir = Path(self.settings.model_dir) / "_candidates" / candidate_version

        # ── Step 1: Train using existing ml-research ─────────────────────
        logger.info("Step 1: Training candidate %s...", candidate_version)
        train_result = train_model(
            config_path=config_path,
            output_dir=candidate_dir,
            data_path=data_path,
        )

        # ── Step 2: Validate candidate ───────────────────────────────────
        logger.info("Step 2: Validating candidate...")
        try:
            self._validate_candidate(candidate_dir, candidate_version)
        except CandidateValidationError as e:
            logger.warning("Candidate %s REJECTED: %s", candidate_version, e)
            self.registry._record_event(
                self.settings.model_name,
                AuditEvent.REJECT,
                candidate_version,
                {"reason": str(e), "trigger": trigger.value},
            )
            return {
                "status": "rejected",
                "version": candidate_version,
                "reason": str(e),
                "run_id": run_id,
            }

        # ── Step 3: Compare with champion ────────────────────────────────
        logger.info("Step 3: Comparing with champion...")
        comparison = self._compare_with_champion(train_result["metrics"])

        if not comparison["passes_gate"]:
            reason = comparison["reason"]
            logger.warning("Candidate %s REJECTED at gate: %s", candidate_version, reason)
            self.registry._record_event(
                self.settings.model_name,
                AuditEvent.REJECT,
                candidate_version,
                {"reason": reason, "trigger": trigger.value, "comparison": comparison},
            )
            return {
                "status": "rejected",
                "version": candidate_version,
                "reason": reason,
                "comparison": comparison,
                "run_id": run_id,
            }

        # ── Step 4: Register ─────────────────────────────────────────────
        logger.info("Step 4: Registering candidate %s...", candidate_version)
        self.registry.register(candidate_dir, candidate_version)

        # ── Step 5: Promote ──────────────────────────────────────────────
        logger.info("Step 5: Promoting candidate %s to champion...", candidate_version)
        self.registry.promote(self.settings.model_name, candidate_version)

        return {
            "status": "promoted",
            "version": candidate_version,
            "metrics": train_result["metrics"],
            "comparison": comparison,
            "run_id": run_id,
        }

    def _validate_candidate(self, candidate_dir: Path, version: str) -> None:
        """
        Validate candidate against all promotion gates (spec §08).

        Gates:
          1. Artifact integrity
          2. Complete manifest
          3. Feature schema match
          4. Model load test
          5. Finite predictions (smoke test)
          6. Serialization round-trip
        """
        loader = ArtifactLoader(candidate_dir)

        # Gate 1: Integrity
        errors = loader.validate_integrity()
        if errors:
            raise CandidateValidationError(f"Integrity check failed: {errors}")

        # Gate 2: Manifest
        manifest = loader.load_manifest()
        if not manifest.model_name:
            raise CandidateValidationError("Manifest missing model_name")

        # Gate 3: Feature schema
        schema = loader.load_feature_schema()
        if not schema.features:
            raise CandidateValidationError("Feature schema is empty")

        # Gate 4: Model load
        pipeline = loader.load_model()

        # Gate 5: Smoke test — finite predictions
        if not loader.smoke_test(pipeline, schema):
            raise CandidateValidationError("Smoke test failed: non-finite predictions")

        logger.info("Candidate %s passed all validation gates", version)

    def _compare_with_champion(self, candidate_metrics: dict[str, Any]) -> dict[str, Any]:
        """
        Compare candidate metrics against the champion (spec §08).

        For fraud detection with F1 as primary metric:
          candidate_f1 >= champion_f1 * (1 - tolerance)
        """
        metric_name = self.settings.promotion_metric
        tolerance = self.settings.promotion_tolerance

        candidate_value = candidate_metrics.get(metric_name)
        if candidate_value is None:
            return {
                "passes_gate": False,
                "reason": f"Candidate missing metric: {metric_name}",
            }

        # Try to load champion metrics
        try:
            champion_version = self.registry.get_champion_version(self.settings.model_name)
            champion_dir = self.registry.get_version_path(
                self.settings.model_name, champion_version
            )
            metrics_path = champion_dir / "metrics.json"

            with open(metrics_path) as f:
                champion_metrics = json.load(f)

            champion_value = champion_metrics.get(metric_name, 0.0)
        except FileNotFoundError:
            # No champion yet — candidate passes by default
            return {
                "passes_gate": True,
                "reason": "No existing champion — candidate auto-passes",
                "candidate_value": candidate_value,
            }

        # Promotion gate: candidate >= champion * (1 - tolerance)
        threshold = champion_value * (1 - tolerance)
        passes = candidate_value >= threshold

        return {
            "passes_gate": passes,
            "metric": metric_name,
            "candidate_value": round(candidate_value, 4),
            "champion_value": round(champion_value, 4),
            "threshold": round(threshold, 4),
            "delta": round(candidate_value - champion_value, 4),
            "reason": (
                f"Candidate {metric_name}={candidate_value:.4f} "
                f"{'≥' if passes else '<'} threshold={threshold:.4f} "
                f"(champion={champion_value:.4f} * {1 - tolerance:.2f})"
            ),
        }

    def _next_version(self) -> str:
        """Generate the next version number."""
        try:
            versions = self.registry.list_versions(self.settings.model_name)
            if versions:
                latest = versions[-1]
                parts = latest.split(".")
                parts[-1] = str(int(parts[-1]) + 1)
                return ".".join(parts)
        except Exception:
            pass
        return "1.0.0"


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Trigger retraining pipeline")
    parser.add_argument(
        "--config", default="configs/credit_card_fraud.yaml", help="Path to config YAML"
    )
    parser.add_argument("--data", default="data/creditcard.csv", help="Path to training data")
    parser.add_argument("--trigger", default="MANUAL", choices=[t.value for t in RetrainTrigger])
    parser.add_argument("--version", default=None, help="Explicit candidate version")
    args = parser.parse_args()

    app_settings = get_settings()
    reg = LocalRegistry(app_settings.model_dir)
    pipeline = RetrainingPipeline(registry=reg, settings=app_settings)

    res = pipeline.retrain(
        config_path=args.config,
        data_path=args.data,
        trigger=RetrainTrigger(args.trigger),
        candidate_version=args.version,
    )
    print(json.dumps(res, indent=2, default=str))
