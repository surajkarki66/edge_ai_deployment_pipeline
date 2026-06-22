from __future__ import annotations

import shutil

from pathlib import Path
from typing import Any


def yolo_export_config(config: dict[str, Any]) -> dict[str, Any]:
    export_cfg = config.get("yolo_export")
    return dict(export_cfg) if isinstance(export_cfg, dict) else {}


def yolo_calibration_yaml(config: dict[str, Any]) -> str | None:
    """Ultralytics dataset yaml for INT8 calibration."""
    export_cfg = yolo_export_config(config)
    raw = export_cfg.get("calibration_data")
    return str(raw) if raw else None


def yolo_export_int8(config: dict[str, Any], *, default_precision: str) -> bool:
    export_cfg = yolo_export_config(config)
    return bool(export_cfg.get("int8", default_precision.lower() == "int8"))


def resolve_imgsz(
    config: dict[str, Any],
    *,
    height: int,
    width: int,
) -> int | tuple[int, int]:
    export_cfg = yolo_export_config(config)
    if export_cfg.get("imgsz") is not None:
        raw = export_cfg["imgsz"]
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return int(raw[0]), int(raw[1])
        return int(raw)
    if height == width:
        return height
    return (height, width)


def stage_weights(source_path: Path, output_dir: Path) -> Path:
    """Copy weights into *output_dir* when Ultralytics must export beside the checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = output_dir / source_path.name
    if weights.resolve() != source_path.resolve():
        shutil.copy2(source_path, weights)
    return weights
