"""
ml-research adapter — spec §04.

Provides the interface between the existing ml-research training
system and the Project 3 operational lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from sklearn.pipeline import Pipeline

from src.ml_research.train import (
    evaluate_model,
    load_model,
    train_model,
)

logger = logging.getLogger(__name__)


# ── Adapter Protocol ─────────────────────────────────────────────────────────
class MLResearchAdapter(Protocol):
    """
    Adapter protocol for the ml-research system (spec §04).

    Project 3 never duplicates training/preprocessing logic.
    """

    def train(self, config_path: str, output_dir: str) -> dict[str, Any]: ...
    def evaluate(self, model_path: str, data_path: str) -> dict[str, Any]: ...
    def load(self, model_path: str) -> Pipeline: ...


# ── Concrete Adapter ────────────────────────────────────────────────────────
class LocalMLResearchAdapter:
    """
    Adapter wrapping the local ml-research training module.

    In a real deployment, this could invoke a CLI subprocess or
    a remote training service. The serving process never calls
    train() directly (spec rule #4).
    """

    def train(
        self,
        config_path: str,
        output_dir: str,
        data_path: str | None = None,
    ) -> dict[str, Any]:
        """Invoke the existing training pipeline."""
        return train_model(config_path, output_dir, data_path)

    def evaluate(
        self,
        model_path: str,
        data_path: str,
        target_column: str = "Class",
        feature_columns: list[str] | None = None,
        drop_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a model on a dataset."""
        return evaluate_model(model_path, data_path, target_column, feature_columns, drop_columns)

    def load(self, model_path: str) -> Pipeline:
        """Load a trained model pipeline."""
        return load_model(model_path)
