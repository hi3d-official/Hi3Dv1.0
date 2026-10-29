"""Verify model snapshot symlinks with deferred transformer loading."""

from pathlib import Path

import pytest

from sparc3d.pipelines.loader import (
    _INITIAL_REQUIRED_FILES,
    _TRANSFORMER_1024_FILES,
    _validate_repository,
)


def _create_initial_snapshot(tmp_path: Path) -> Path:
    """Create the initial model artifacts needed for deferred loading."""
    root = tmp_path / "snapshots" / "revision"
    for relative in _INITIAL_REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "", encoding="utf-8")
    return root


def test_cached_deferred_symlinks_are_allowed(tmp_path: Path) -> None:
    """Accept previously cached high-resolution artifacts during initial loading."""
    root = _create_initial_snapshot(tmp_path)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for relative in _TRANSFORMER_1024_FILES:
        blob = blobs / relative.replace("/", "_")
        blob.touch()
        link = root / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)

    _validate_repository(root, required_files=_INITIAL_REQUIRED_FILES)


def test_unexpected_symlink_is_still_rejected(tmp_path: Path) -> None:
    """Reject symlinks outside the complete model artifact allowlist."""
    root = _create_initial_snapshot(tmp_path)
    target = tmp_path / "unexpected.txt"
    target.touch()
    (root / "unexpected.txt").symlink_to(target)

    with pytest.raises(ValueError, match="unexpected symlink"):
        _validate_repository(root, required_files=_INITIAL_REQUIRED_FILES)


def test_missing_initial_artifact_is_still_rejected(tmp_path: Path) -> None:
    """Continue to require all initial model artifacts."""
    root = _create_initial_snapshot(tmp_path)
    (root / "vae" / "config.json").unlink()

    with pytest.raises(FileNotFoundError, match="required model artifact"):
        _validate_repository(root, required_files=_INITIAL_REQUIRED_FILES)
