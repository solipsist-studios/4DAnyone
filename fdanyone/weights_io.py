"""One reader for model weights, whether they ship as ``.safetensors`` or ``.pth``.

The same tensors are published in both formats (for example the Wan2.2 VAE as
``Wan2.2_VAE.pth`` and as ComfyUI's ``wan2.2_vae.safetensors``), so every loader
that accepts either goes through :func:`load_state_dict`.

``.pth`` files are read with ``torch.load(weights_only=True)`` only; a pickle that
needs arbitrary code is refused rather than executed.
"""

from __future__ import annotations

from pathlib import Path

from fdanyone.errors import AssetError

SAFETENSORS_SUFFIXES = (".safetensors",)
TORCH_SUFFIXES = (".pth", ".pt", ".bin")
WEIGHT_SUFFIXES = SAFETENSORS_SUFFIXES + TORCH_SUFFIXES

# Some checkpoints nest the tensors under this key next to training metadata.
WRAPPER_KEY = "model_state"


def is_weights_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in WEIGHT_SUFFIXES


def _keep(key: str, include_prefix: str | None, exclude_prefixes: tuple[str, ...]) -> bool:
    return (include_prefix is None or key.startswith(include_prefix)) and not any(
        key.startswith(prefix) for prefix in exclude_prefixes
    )


def _read_safetensors(path: Path, include_prefix, exclude_prefixes):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise AssetError("safetensors is required to load .safetensors weights.") from exc
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        keys = [key for key in checkpoint.keys() if _keep(key, include_prefix, exclude_prefixes)]
        return (
            {key: checkpoint.get_tensor(key) for key in keys},
            dict(checkpoint.metadata() or {}),
        )


def _read_torch(path: Path, include_prefix, exclude_prefixes):
    import torch

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict) and isinstance(loaded.get(WRAPPER_KEY), dict):
        loaded = loaded[WRAPPER_KEY]
    if not isinstance(loaded, dict):
        raise AssetError(f"{path} does not hold a state dict (got {type(loaded).__name__}).")
    return ({key: value for key, value in loaded.items() if _keep(key, include_prefix, exclude_prefixes)}, {})


def load_state_dict(
    path: str | Path,
    *,
    include_prefix: str | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> tuple[dict, dict]:
    """Return ``(state_dict, metadata)`` from a weights file, chosen by extension.

    ``.safetensors`` is read tensor by tensor, so a prefix filter never loads the
    tensors it drops. ``.pth``/``.pt``/``.bin`` are read with
    ``torch.load(map_location="cpu", weights_only=True)`` and a ``model_state``
    wrapper is unwrapped. Tensors keep the dtype stored in the file. ``metadata`` is
    the safetensors header metadata, and ``{}`` for torch files.
    """

    resolved = Path(path).expanduser()
    suffix = resolved.suffix.lower()
    if suffix not in WEIGHT_SUFFIXES:
        raise AssetError(
            f"Unsupported weights file {resolved}: expected one of {', '.join(WEIGHT_SUFFIXES)}."
        )
    if not resolved.is_file():
        raise AssetError(f"Weights file does not exist: {resolved}")
    exclude_prefixes = tuple(exclude_prefixes)
    try:
        if suffix in SAFETENSORS_SUFFIXES:
            return _read_safetensors(resolved, include_prefix, exclude_prefixes)
        return _read_torch(resolved, include_prefix, exclude_prefixes)
    except AssetError:
        raise
    except Exception as exc:
        raise AssetError(f"Could not read weights from {resolved}: {exc}") from exc
