# PyTorch → ONNX export helpers (adapted from Ultralytics).
# Copyright (C) Ultralytics Inc.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import logging
import torch

from pathlib import Path
from typing import Any

from edge_ai_deployment_pipeline.adapters.jetson.toolchains.onnx2engine import onnx2engine
from edge_ai_deployment_pipeline.adapters.jetson.toolchains.torch2onnx import export_onnx
from edge_ai_deployment_pipeline.adapters.jetson.toolchains.trt_calibration import TensorRTCalibrationLoader

logger = logging.getLogger(__name__)


def _build_calibration_loader(
    calibration_data: Path | str,
    shape: tuple[int, int, int, int],
    *,
    channels_last: bool = True,
    batch_size: int = 1,
    device: str = "cuda:0",
) -> TensorRTCalibrationLoader:
    return TensorRTCalibrationLoader(
        calibration_data,
        shape,
        channels_last=channels_last,
        batch_size=batch_size,
        device=device,
    )


def export_engine(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_dir: Path | str,
    *,
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
    metadata: dict[str, Any] | None = None,
    verbose: bool = False,
    device: str | torch.device = "cuda:0",
    calibration_data: Path | str | None = None,
    channels_last: bool = False,
    calib_batch_size: int = 1,
) -> tuple[str, str]:
    """Export a PyTorch model to TensorRT (ONNX intermediate, then engine).

    Returns:
        (onnx_path, engine_path) as strings.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TensorRT export requires a CUDA GPU (use a Jetson or machine with CUDA)."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)
    if im.device.type == "cpu":
        im = im.to(dev)
    model = model.to(dev)

    onnx_path = output_dir / "model.onnx"
    engine_path = output_dir / "model.engine"

    logger.info("Exporting ONNX to %s", onnx_path)
    export_onnx(
        model,
        onnx_path,
        im,
        dynamic=dynamic,
        simplify=simplify,
        convert_onnx_fp16=False,
    )

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX export failed: {onnx_path}")

    shape = tuple(int(d) for d in im.shape)
    if len(shape) != 4:
        raise ValueError(f"Expected NCHW example input (1,C,H,W), got shape {shape}")

    calibrator = None
    if int8:
        if not calibration_data:
            raise ValueError("calibration_data is required for INT8 TensorRT export")
        calibrator = _build_calibration_loader(
            calibration_data,
            shape,
            channels_last=channels_last,
            batch_size=calib_batch_size,
            device=str(device),
        )

    logger.info("Building TensorRT engine at %s", engine_path)
    onnx2engine(
        str(onnx_path),
        engine_path,
        half=half,
        int8=int8,
        dynamic=dynamic,
        shape=shape,
        dataset=calibrator,
        metadata=metadata,
        verbose=verbose,
        prefix="TensorRT: ",
    )

    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT export failed: {engine_path}")

    return str(onnx_path.resolve()), str(engine_path.resolve())


def export_engine_from_onnx(
    onnx_path: Path | str,
    output_dir: Path | str,
    *,
    trt_shape: tuple[int, int, int, int],
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    metadata: dict[str, Any] | None = None,
    verbose: bool = False,
    calibration_data: Path | str | None = None,
    channels_last: bool = True,
    calib_batch_size: int = 1,
) -> tuple[str, str]:
    """Build a TensorRT engine from an existing ONNX file."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TensorRT export requires a CUDA GPU (use a Jetson or machine with CUDA)."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    engine_path = output_dir / "model.engine"
    dest_onnx = output_dir / "model.onnx"
    if onnx_path.resolve() != dest_onnx.resolve():
        import shutil

        shutil.copy2(onnx_path, dest_onnx)
        onnx_path = dest_onnx

    calibrator = None
    if int8:
        if not calibration_data:
            raise ValueError("calibration_data is required for INT8 TensorRT export")
        calibrator = _build_calibration_loader(
            calibration_data,
            trt_shape,
            channels_last=channels_last,
            batch_size=calib_batch_size,
        )

    logger.info("Building TensorRT engine from ONNX at %s", engine_path)
    onnx2engine(
        str(onnx_path),
        engine_path,
        half=half,
        int8=int8,
        dynamic=dynamic,
        shape=trt_shape,
        dataset=calibrator,
        metadata=metadata,
        verbose=verbose,
        prefix="TensorRT: ",
    )

    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT export failed: {engine_path}")

    return str(onnx_path.resolve()), str(engine_path.resolve())
