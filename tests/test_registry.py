"""
Unit tests — registry (spec §12).

Tests:
  - register version
  - champion pointer
  - promotion
  - rollback
  - immutability
  - audit history
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import AuditEvent
from src.registry.local_registry import LocalRegistry

pytestmark = pytest.mark.unit


class TestLocalRegistry:
    """Test filesystem-based model registry."""

    def test_register_version(self, tmp_dir: Path, mock_model_dir: Path) -> None:
        registry = LocalRegistry(tmp_dir / "registry")
        manifest = registry.register(mock_model_dir, "1.0.0")
        assert manifest.model_name == "credit-card-fraud"

    def test_list_versions(self, tmp_dir: Path, mock_model_dir: Path) -> None:
        registry = LocalRegistry(tmp_dir / "registry")
        registry.register(mock_model_dir, "1.0.0")
        versions = registry.list_versions("credit-card-fraud")
        assert "1.0.0" in versions

    def test_promote_to_champion(self, tmp_dir: Path, mock_model_dir: Path) -> None:
        registry = LocalRegistry(tmp_dir / "registry")
        registry.register(mock_model_dir, "1.0.0")
        registry.promote("credit-card-fraud", "1.0.0")
        assert registry.get_champion_version("credit-card-fraud") == "1.0.0"

    def test_rollback(self, tmp_dir: Path, mock_model_dir: Path) -> None:
        registry = LocalRegistry(tmp_dir / "registry")
        registry.register(mock_model_dir, "1.0.0")
        registry.promote("credit-card-fraud", "1.0.0")

        # Register and promote v2
        registry.register(mock_model_dir, "1.1.0")
        registry.promote("credit-card-fraud", "1.1.0")
        assert registry.get_champion_version("credit-card-fraud") == "1.1.0"

        # Rollback
        rolled_back = registry.rollback("credit-card-fraud")
        assert rolled_back == "1.0.0"
        assert registry.get_champion_version("credit-card-fraud") == "1.0.0"

    def test_cannot_register_duplicate(self, tmp_dir: Path, mock_model_dir: Path) -> None:
        registry = LocalRegistry(tmp_dir / "registry")
        registry.register(mock_model_dir, "1.0.0")
        with pytest.raises(ValueError, match="already registered"):
            registry.register(mock_model_dir, "1.0.0")

    def test_audit_history(self, tmp_dir: Path, mock_model_dir: Path) -> None:
        registry = LocalRegistry(tmp_dir / "registry")
        registry.register(mock_model_dir, "1.0.0")
        registry.promote("credit-card-fraud", "1.0.0")

        history = registry.get_history("credit-card-fraud")
        events = [h["event"] for h in history]
        assert AuditEvent.REGISTER.value in events
        assert AuditEvent.PROMOTE.value in events

    def test_no_champion_raises(self, tmp_dir: Path) -> None:
        registry = LocalRegistry(tmp_dir / "registry")
        with pytest.raises(FileNotFoundError):
            registry.get_champion_version("nonexistent")
