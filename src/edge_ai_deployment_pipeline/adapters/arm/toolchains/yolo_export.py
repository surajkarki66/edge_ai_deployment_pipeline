from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from edge_ai_deployment_pipeline.core.ultralytics_export import (
    resolve_imgsz,
    stage_weights,
    yolo_export_config,
)

logger = logging.getLogger(__name__)


def _ncnn_export_kwargs(config: dict[str, Any], *, imgsz: int | tuple[int, int]) -> dict[str, Any]:
    export_cfg = yolo_export_config(config)
    kwargs: dict[str, Any] = {
        "format": str(export_cfg.get("format", "ncnn")),
        "imgsz": imgsz,
        "half": bool(export_cfg.get("half", False)),
        "batch": int(export_cfg.get("batch", 1)),
    }
    device = export_cfg.get("device")
    if device:
        kwargs["device"] = str(device)
    return kwargs


def _find_ncnn_outputs(ncnn_dir: Path) -> tuple[Path, Path, Path | None]:
    param_path = ncnn_dir / "model.ncnn.param"
    bin_path = ncnn_dir / "model.ncnn.bin"
    py_path = ncnn_dir / "model_ncnn.py"
    if not param_path.is_file() or not bin_path.is_file():
        raise FileNotFoundError(
            f"YOLO NCNN export did not produce model.ncnn.param/.bin in {ncnn_dir}"
        )
    return param_path, bin_path, py_path if py_path.is_file() else None


def export_yolo_to_ncnn(
    source_path: Path,
    output_dir: Path,
    *,
    height: int,
    width: int,
    config: dict[str, Any],
) -> tuple[Path, Path, Path | None, Path]:
    """Export a YOLO checkpoint to NCNN via Ultralytics."""
    from ultralytics import YOLO

    imgsz = resolve_imgsz(config, height=height, width=width)
    export_kwargs = _ncnn_export_kwargs(config, imgsz=imgsz)
    weights = stage_weights(source_path, output_dir)

    logger.info(
        "Exporting YOLO %s → %s (imgsz=%s, half=%s)",
        weights,
        export_kwargs["format"],
        imgsz,
        export_kwargs["half"],
    )
    model = YOLO(str(weights.resolve()))
    ncnn_dir = Path(str(model.export(**export_kwargs))).resolve()
    weights.with_suffix(".torchscript").unlink(missing_ok=True)

    param_dest, bin_dest, py_dest = _find_ncnn_outputs(ncnn_dir)
    logger.info("NCNN model directory: %s", ncnn_dir)
    return param_dest, bin_dest, py_dest, ncnn_dir
