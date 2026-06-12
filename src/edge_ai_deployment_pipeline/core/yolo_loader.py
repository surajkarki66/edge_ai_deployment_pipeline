from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yolo(model_path: str | Path) -> Any:
    """Load an Ultralytics YOLO checkpoint."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLO (.pt) checkpoints. "
            "Install with: pip install ultralytics"
        ) from exc

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {path}")
    return YOLO(str(path.resolve()))


def yolo_class_names(model: Any) -> list[str]:
    """Return human-readable class names from an Ultralytics YOLO model."""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return ["object"]
