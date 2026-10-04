"""
Local filesystem model registry — spec §06.

Provides a zero-infrastructure registry for the demo:

    models/credit-card-fraud/1.0.0/
    models/credit-card-fraud/1.1.0/
    models/credit-card-fraud/champion.json

A registry interface is maintained so the serving code is not
coupled to filesystem semantics.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from src.artifact.contract import ModelManifest
from src.artifact.loader import ArtifactLoader
from src.config import AuditEvent

logger = logging.getLogger(__name__)


# ── Registry Protocol ────────────────────────────────────────────────────────
class ModelRegistry(Protocol):
    """Interface so serving code is not coupled to registry implementation."""

    def register(self, artifact_dir: str | Path, version: str) -> ModelManifest: ...
    def get_champion_version(self, model_name: str) -> str: ...
    def get_version_path(self, model_name: str, version: str) -> Path: ...
    def promote(self, model_name: str, version: str) -> None: ...
    def rollback(self, model_name: str) -> str: ...
    def list_versions(self, model_name: str) -> list[str]: ...


# ── Local Filesystem Registry ────────────────────────────────────────────────
class LocalRegistry:
    """
    Filesystem-based model registry.

    Directory layout:
        {base_dir}/{model_name}/{version}/  — immutable artifact bundles
        {base_dir}/{model_name}/champion.json — champion pointer
        {base_dir}/{model_name}/history.json  — promotion/rollback audit
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, model_name: str) -> Path:
        d = self.base_dir / model_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _champion_file(self, model_name: str) -> Path:
        return self._model_dir(model_name) / "champion.json"

    def _history_file(self, model_name: str) -> Path:
        return self._model_dir(model_name) / "history.json"

    def _load_history(self, model_name: str) -> list[dict[str, Any]]:
        hf = self._history_file(model_name)
        if hf.exists():
            with open(hf) as f:
                return cast(list[dict[str, Any]], json.load(f))
        return []

    def _save_history(self, model_name: str, history: list[dict[str, Any]]) -> None:
        with open(self._history_file(model_name), "w") as f:
            json.dump(history, f, indent=2)

    def _record_event(
        self,
        model_name: str,
        event: AuditEvent,
        version: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        history = self._load_history(model_name)
        history.append(
            {
                "event": event.value,
                "model_name": model_name,
                "version": version,
                "timestamp": datetime.now(UTC).isoformat(),
                "details": details or {},
            }
        )
        self._save_history(model_name, history)

    # ── Public API ───────────────────────────────────────────────────────
    def register(self, artifact_dir: str | Path, version: str) -> ModelManifest:
        """
        Register a new model version by copying artifacts into the registry.

        Artifacts are immutable once registered (spec §06).
        """
        artifact_dir = Path(artifact_dir)
        loader = ArtifactLoader(artifact_dir)
        bundle = loader.load_bundle()

        model_name = bundle.manifest.model_name
        version_dir = self._model_dir(model_name) / version

        if version_dir.exists():
            raise ValueError(f"Version {version} already registered for {model_name}")

        # Copy artifacts (immutable)
        shutil.copytree(artifact_dir, version_dir)
        logger.info("Registered %s v%s at %s", model_name, version, version_dir)

        self._record_event(model_name, AuditEvent.REGISTER, version)
        return bundle.manifest

    def get_champion_version(self, model_name: str) -> str:
        """Get the current champion version."""
        cf = self._champion_file(model_name)
        if not cf.exists():
            raise FileNotFoundError(f"No champion set for {model_name}")
        with open(cf) as f:
            data = json.load(f)
        return str(data["version"])

    def get_version_path(self, model_name: str, version: str) -> Path:
        """Get the filesystem path for a specific version."""
        vp = self._model_dir(model_name) / version
        if not vp.is_dir():
            raise FileNotFoundError(f"Version {version} not found for {model_name}")
        return vp

    def promote(self, model_name: str, version: str) -> None:
        """
        Promote a version to champion.

        This changes the champion pointer, not the Docker image (spec rule #10).
        """
        # Verify version exists
        version_dir = self.get_version_path(model_name, version)

        # Load and validate before promotion
        loader = ArtifactLoader(version_dir)
        bundle = loader.load_bundle()
        pipeline = loader.load_model()

        if not loader.smoke_test(pipeline, bundle.feature_schema):
            raise ValueError(f"Smoke test failed for {model_name} v{version}")

        # Read previous champion for history
        previous_version = None
        with contextlib.suppress(FileNotFoundError):
            previous_version = self.get_champion_version(model_name)

        # Update champion pointer
        champion_data = {
            "model_name": model_name,
            "version": version,
            "promoted_at": datetime.now(UTC).isoformat(),
            "previous_version": previous_version,
        }
        with open(self._champion_file(model_name), "w") as f:
            json.dump(champion_data, f, indent=2)

        logger.info(
            "Promoted %s v%s to champion (previous: %s)", model_name, version, previous_version
        )
        self._record_event(
            model_name,
            AuditEvent.PROMOTE,
            version,
            {"previous_version": previous_version},
        )

    def rollback(self, model_name: str) -> str:
        """
        Rollback to the previous known-good version.

        Changes the champion pointer — does not delete versions (spec §06).
        """
        cf = self._champion_file(model_name)
        if not cf.exists():
            raise FileNotFoundError(f"No champion to rollback for {model_name}")

        with open(cf) as f:
            data = json.load(f)

        previous = data.get("previous_version")
        if not previous:
            raise ValueError(f"No previous version to rollback to for {model_name}")

        current = data["version"]

        # Rollback = promote previous
        self.promote(model_name, previous)
        logger.info("Rolled back %s from v%s to v%s", model_name, current, previous)
        self._record_event(
            model_name,
            AuditEvent.ROLLBACK,
            previous,
            {"rolled_back_from": current},
        )
        return str(previous)

    def list_versions(self, model_name: str) -> list[str]:
        """List all registered versions."""
        model_dir = self._model_dir(model_name)
        versions = []
        for child in sorted(model_dir.iterdir()):
            if child.is_dir() and (child / "manifest.json").exists():
                versions.append(child.name)
        return versions

    def get_history(self, model_name: str) -> list[dict[str, Any]]:
        """Get the full audit history for a model."""
        return self._load_history(model_name)


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    from src.config import get_settings

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Model registry CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List registered versions")
    p_list.add_argument("--model", default="credit-card-fraud")

    # champion
    p_champ = subparsers.add_parser("champion", help="Show current champion")
    p_champ.add_argument("--model", default="credit-card-fraud")

    # register
    p_reg = subparsers.add_parser("register", help="Register a model version")
    p_reg.add_argument("--dir", required=True, help="Directory containing artifact bundle")
    p_reg.add_argument("--version", required=True, help="Version string (e.g. 1.0.0)")

    # promote
    p_prom = subparsers.add_parser("promote", help="Promote a version to champion")
    p_prom.add_argument("--model", default="credit-card-fraud")
    p_prom.add_argument("--version", required=True)

    # rollback
    p_roll = subparsers.add_parser("rollback", help="Rollback champion to previous version")
    p_roll.add_argument("--model", default="credit-card-fraud")

    # history
    p_hist = subparsers.add_parser("history", help="Show audit history")
    p_hist.add_argument("--model", default="credit-card-fraud")

    args = parser.parse_args()
    app_settings = get_settings()
    reg = LocalRegistry(app_settings.model_dir)

    if args.command == "list":
        versions = reg.list_versions(args.model)
        print(f"Registered versions for {args.model}: {versions}")
    elif args.command == "champion":
        try:
            champ = reg.get_champion_version(args.model)
            print(f"Current champion for {args.model}: {champ}")
        except FileNotFoundError:
            print(f"No champion set for {args.model}")
    elif args.command == "register":
        manifest = reg.register(args.dir, args.version)
        print(f"Registered {manifest.model_name} v{manifest.model_version}")
    elif args.command == "promote":
        reg.promote(args.model, args.version)
        print(f"Promoted {args.model} v{args.version} to champion")
    elif args.command == "rollback":
        prev = reg.rollback(args.model)
        print(f"Rolled back {args.model} to v{prev}")
    elif args.command == "history":
        history = reg.get_history(args.model)
        print(json.dumps(history, indent=2))
