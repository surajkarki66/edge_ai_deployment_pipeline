from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DPU_PATCHED = False
_DPU_KEY_MARKERS = ("ch_select_a", "ch_select_b")
_PSABLOCK_DPU_MARKER = ".m.0.conv1."


def _invalidate_ultralytics_import_cache() -> None:
    """Drop cached ultralytics modules so patched block/head are re-imported."""
    for name in list(sys.modules):
        if name == "ultralytics" or name.startswith("ultralytics."):
            del sys.modules[name]


def ensure_ultralytics_dpu_patch() -> None:
    """Copy DPU-compatible head/block into the installed ultralytics package."""
    global _DPU_PATCHED
    if _DPU_PATCHED:
        return

    from edge_ai_deployment_pipeline.adapters.fpga.toolchains.ultralytics_dpu.patch_ultralytics import (
        patch_ultralytics,
    )

    patch_ultralytics()
    _invalidate_ultralytics_import_cache()
    _DPU_PATCHED = True
    logger.info("Applied Ultralytics DPU custom layers (head.py, block.py) on host")


def checkpoint_needs_dpu_custom_layers(model_path: str | Path) -> bool:
    """Return True when a checkpoint was saved with DPU-patched ultralytics layers."""
    import torch

    path = Path(model_path)
    try:
        checkpoint = torch.load(str(path.resolve()), map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.debug("Could not inspect checkpoint for DPU markers (%s): %s", path.name, exc)
        return False

    model = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if model is None:
        return False

    try:
        state_dict = model.state_dict() if hasattr(model, "state_dict") else {}
    except Exception:
        state_dict = {}

    for key in state_dict:
        if any(marker in key for marker in _DPU_KEY_MARKERS):
            return True
        if _PSABLOCK_DPU_MARKER in key:
            return True

    for module in model.modules():
        module_name = type(module).__name__
        if module_name == "C2f" and hasattr(module, "ch_select_a"):
            return True
        if module_name == "PSABlock" and hasattr(module, "conv1"):
            return True

    return False


def load_yolo(model_path: str | Path, *, patch_dpu_layers: bool = False) -> Any:
    """Load an Ultralytics YOLO checkpoint.

    Set *patch_dpu_layers* to apply Vitis-AI DPU custom layers (``head.py``,
    ``block.py``) before loading. This must only be enabled for the Vitis-AI
    FPGA flow — other targets (e.g. STM32) require stock ultralytics modules.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {path}")

    if not patch_dpu_layers and checkpoint_needs_dpu_custom_layers(path):
        raise ValueError(
            f"Checkpoint {path.name} was saved with Vitis-AI DPU custom Ultralytics "
            "layers. Re-run with --board-id zcu104 (or another FPGA board) or "
            "--toolchain vitis_ai so those layers are patched before loading."
        )

    if patch_dpu_layers:
        ensure_ultralytics_dpu_patch()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLO (.pt) checkpoints. "
            "Install with: pip install ultralytics"
        ) from exc

    return YOLO(str(path.resolve()))


def yolo_class_names(model: Any) -> list[str]:
    """Return human-readable class names from an Ultralytics YOLO model."""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return ["object"]
