"""Safe local and Hugging Face artifact loading for the public pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "vae/config.json",
    "vae/model.safetensors",
    "transformer_512/config.json",
    "transformer_512/model.safetensors",
    "transformer_1024/config.json",
    "transformer_1024/model.safetensors",
)
"""Complete allowlist of files consumed from a model repository."""

_TRANSFORMER_1024_FILES = (
    "transformer_1024/config.json",
    "transformer_1024/model.safetensors",
)
"""Files deferred until the first high-resolution pass for Hub repositories."""

_INITIAL_REQUIRED_FILES = tuple(
    path for path in _REQUIRED_FILES if path not in _TRANSFORMER_1024_FILES
)
"""Files required to construct a pipeline before its high-resolution pass."""

_UNSAFE_SUFFIXES = {".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth"}
"""Checkpoint suffixes that may invoke pickle-based deserialization."""

_DEFAULT_SCHEDULER_CLASS = "FlowMatchHeunScheduler"
"""Integrator used when a scheduler config omits ``_class_name``."""

_INDEX_PATHS = {
    "scheduler": "scheduler",
    "vae": "vae",
    "transformer_512": "transformer_512",
    "transformer_1024": "transformer_1024",
}
"""Optional model-index path declarations and their only accepted values."""


def resolve_model_directory(
    pretrained_model_name_or_path: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    local_files_only: bool,
    defer_transformer_1024: bool = False,
) -> Path:
    """Resolve a local model directory or an allowlisted Hugging Face snapshot.

    Args:
        pretrained_model_name_or_path: Local directory or Hugging Face repository ID.
        revision: Optional model-repository revision.
        cache_dir: Optional Hugging Face cache directory.
        local_files_only: Whether Hub resolution must use cached files only.
        defer_transformer_1024: Whether a Hub download should omit high-resolution
            transformer files until first use.

    Returns:
        Validated local artifact root.

    Raises:
        FileNotFoundError: A path-like local source does not exist.
        ValueError: The source is a file, a ModelScope remote URI, or unsafe.
    """
    source = str(pretrained_model_name_or_path)
    candidate = Path(source).expanduser()
    if candidate.is_dir():
        root = candidate.resolve()
        required_files = _REQUIRED_FILES
    elif candidate.exists():
        raise ValueError(f"model source must be a directory, got {candidate}")
    elif source.startswith("modelscope://"):
        raise ValueError(
            "ModelScope network loading is unsupported; pass a downloaded local directory"
        )
    elif candidate.is_absolute() or source.startswith((".", "~")):
        raise FileNotFoundError(f"local model directory does not exist: {candidate}")
    else:
        from huggingface_hub import snapshot_download

        required_files = (
            _INITIAL_REQUIRED_FILES if defer_transformer_1024 else _REQUIRED_FILES
        )
        root = Path(
            snapshot_download(
                repo_id=source,
                revision=revision,
                cache_dir=None if cache_dir is None else str(cache_dir),
                local_files_only=local_files_only,
                allow_patterns=list(required_files),
            )
        ).resolve()
    _validate_repository(root, required_files=required_files)
    return root


def load_deferred_transformer_1024(
    root: Path,
    pretrained_model_name_or_path: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> Any:
    """Download a deferred Hub artifact if needed, then strict-load it on CPU."""
    source = str(pretrained_model_name_or_path)
    if not Path(source).expanduser().is_dir():
        from huggingface_hub import snapshot_download

        resolved_revision = _resolved_snapshot_revision(root, revision)
        downloaded_root = Path(
            snapshot_download(
                repo_id=source,
                revision=resolved_revision,
                cache_dir=None if cache_dir is None else str(cache_dir),
                local_files_only=local_files_only,
                allow_patterns=list(_TRANSFORMER_1024_FILES),
            )
        ).resolve()
        if downloaded_root != root:
            raise RuntimeError(
                "deferred transformer resolved to a different model snapshot"
            )
    _validate_repository(root, required_files=_REQUIRED_FILES)
    return load_transformer(root, "transformer_1024")


def load_scheduler(root: Path) -> Any:
    """Load the allowlisted scheduler configuration from ``root``."""
    from sparc3d.schedulers.flow_match_euler import FlowMatchEulerScheduler
    from sparc3d.schedulers.flow_match_heun import FlowMatchHeunScheduler

    schedulers = {
        "FlowMatchHeunScheduler": FlowMatchHeunScheduler,
        "FlowMatchEulerScheduler": FlowMatchEulerScheduler,
    }
    values = _read_json(root / "scheduler" / "scheduler_config.json")
    class_name = values.pop("_class_name", _DEFAULT_SCHEDULER_CLASS)
    if class_name not in schedulers:
        raise ValueError(f"unsupported scheduler class {class_name!r}")
    allowed = {"num_inference_steps", "sigma_min", "time_rescale_factor"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown scheduler config keys: {unknown}")
    return schedulers[class_name](**values)


def load_conditioner(
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
    dino_model_id: str | Path | None = None,
    clip_model_id: str | Path | None = None,
) -> Any:
    """Load the fixed shared image conditioner on the CPU.

    Args:
        cache_dir: Optional Hugging Face cache directory.
        local_files_only: Whether encoder loading must use cached files only.
        dino_model_id: Optional DINOv2 repository ID or local directory,
            overriding the default encoder repository.
        clip_model_id: Optional CLIP repository ID or local directory.

    Returns:
        Frozen fp32 conditioner in evaluation mode on the CPU.
    """
    from sparc3d.models.conditioners.image import ImageConditioner

    overrides: dict[str, str | Path] = {}
    if dino_model_id is not None:
        overrides["dino_model_id"] = dino_model_id
    if clip_model_id is not None:
        overrides["clip_model_id"] = clip_model_id

    return (
        ImageConditioner(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **overrides,
        )
        .to("cpu")
        .eval()
    )


def load_transformer(root: Path, artifact_name: str) -> Any:
    """Load one strict mixed-precision transformer artifact on the CPU.

    Args:
        root: Validated artifact repository.
        artifact_name: ``"transformer_512"`` or ``"transformer_1024"``.

    Returns:
        Transformer in evaluation mode on the CPU.

    Raises:
        ValueError: ``artifact_name`` is not allowlisted.
        RuntimeError: Checkpoint keys do not match the model.
    """
    if artifact_name not in {"transformer_512", "transformer_1024"}:
        raise ValueError(f"unsupported transformer artifact {artifact_name!r}")
    from safetensors.torch import load_file
    from sparc3d.models.transformers import (
        Sparc3DTransformer,
        Sparc3DTransformerConfig,
    )

    artifact = root / artifact_name
    config = Sparc3DTransformerConfig.from_dict(_read_json(artifact / "config.json"))
    model = Sparc3DTransformer(config)
    model.load_state_dict(
        load_file(str(artifact / "model.safetensors"), device="cpu"), strict=True
    )
    return model.to("cpu").eval()


def load_vae(root: Path) -> Any:
    """Load the strict sparse VAE artifact on the CPU."""
    from sparc3d.models.autoencoders.config import load_sparconv_vae

    return load_sparconv_vae(root / "vae", device="cpu")


def _validate_repository(
    root: Path, *, required_files: tuple[str, ...] = _REQUIRED_FILES
) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {root}")
    allowed_symlink_paths = {Path(relative) for relative in _REQUIRED_FILES}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink() and relative not in allowed_symlink_paths:
            raise ValueError(f"model repository has an unexpected symlink: {path}")
        if path.is_file() and path.suffix.lower() in _UNSAFE_SUFFIXES:
            raise ValueError(f"unsafe checkpoint file is not permitted: {path}")
    for relative in required_files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required model artifact is missing: {path}")

    index = _read_json(root / "model_index.json")
    allowed = {"_class_name", "_format_version", "_sparc3d_version", *_INDEX_PATHS}
    unknown = sorted(set(index) - allowed)
    if unknown:
        raise ValueError(f"unknown model index keys: {unknown}")
    class_name = index.get("_class_name", "Sparc3DPipeline")
    if class_name != "Sparc3DPipeline":
        raise ValueError(f"unsupported pipeline class {class_name!r}")
    for key, expected in _INDEX_PATHS.items():
        if key in index and index[key] != expected:
            raise ValueError(
                f"model index entry {key!r} must be the fixed path {expected!r}"
            )
    for key in ("_format_version", "_sparc3d_version"):
        if key in index and not isinstance(index[key], (str, int, float)):
            raise ValueError(f"model index metadata {key!r} must be scalar")


def _resolved_snapshot_revision(root: Path, requested_revision: str | None) -> str | None:
    parent = root.parent
    if parent.name == "snapshots":
        return root.name
    return requested_revision


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(values, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return dict(values)
