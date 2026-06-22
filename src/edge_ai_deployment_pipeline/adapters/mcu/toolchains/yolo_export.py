from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

from edge_ai_deployment_pipeline.core.ultralytics_export import (
    resolve_imgsz,
    stage_weights,
    yolo_calibration_yaml,
    yolo_export_config,
    yolo_export_int8,
)

logger = logging.getLogger(__name__)


def _tflite_export_kwargs(
    config: dict[str, Any],
    *,
    imgsz: int | tuple[int, int],
    precision: str,
) -> dict[str, Any]:
    export_cfg = yolo_export_config(config)
    int8 = yolo_export_int8(config, default_precision=precision)

    kwargs: dict[str, Any] = {
        "format": str(export_cfg.get("format", "tflite")),
        "imgsz": imgsz,
        "int8": int8,
    }

    workspace = export_cfg.get("workspace")
    if workspace is not None:
        kwargs["workspace"] = workspace

    calibration_yaml = yolo_calibration_yaml(config)
    if calibration_yaml:
        kwargs["data"] = calibration_yaml
    elif int8:
        raise ValueError(
            "yolo_export.calibration_data (Ultralytics dataset yaml) is required "
            "for TensorRT INT8 export"
        )

    fraction = export_cfg.get("fraction")
    if fraction is not None:
        kwargs["fraction"] = float(fraction)

    device = export_cfg.get("device")
    if device:
        kwargs["device"] = str(device)
    else:
        kwargs["device"] = "0"

    return kwargs


def export_yolo_to_tflite(
    source_path: Path,
    output_dir: Path,
    *,
    height: int,
    width: int,
    config: dict[str, Any],
    precision: str,
) -> Path:
    """Export a YOLO checkpoint to tflite format via Ultralytics."""
    from ultralytics import YOLO

    imgsz = resolve_imgsz(config, height=height, width=width)
    export_kwargs = _tflite_export_kwargs(config, imgsz=imgsz, precision=precision)
    weights = stage_weights(source_path, output_dir)

    logger.info(
        "Exporting YOLO %s → TFlite (imgsz=%s, int8=%s, half=%s, device=%s)",
        weights,
        imgsz,
        export_kwargs["int8"],
        "", "")
    model = YOLO(str(weights.resolve()))
    tflite_path = Path(str(model.export(**export_kwargs))).resolve()

    for suffix in (".onnx", ".torchscript"):
        weights.with_suffix(suffix).unlink(missing_ok=True)

    if not tflite_path.is_file():
        raise FileNotFoundError(f"YOLO TFlite export did not produce the file: {tflite_path}")

    logger.info("TFlite Model: %s", tflite_path)
    return tflite_path
