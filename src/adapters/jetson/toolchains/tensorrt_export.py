"""PyTorch → ONNX → TensorRT engine export.

Adapted from Ultralytics ``export_engine`` / ``onnx2engine`` flow.
See: https://github.com/ultralytics/ultralytics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from adapters.jetson.toolchains.onnx2engine import onnx2engine
from adapters.jetson.toolchains.torch2onnx import export_onnx

logger = logging.getLogger(__name__)


def export_engine(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_dir: Path | str,
    *,
    workspace: int | None = None,
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
    metadata: dict[str, Any] | None = None,
    verbose: bool = False,
    device: str | torch.device = "cuda:0",
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

    if int8:
        raise ValueError(
            "INT8 TensorRT export requires a calibration dataset; not supported in the pipeline yet."
        )

    shape = tuple(int(d) for d in im.shape)
    if len(shape) != 4:
        raise ValueError(f"Expected NCHW example input (1,C,H,W), got shape {shape}")

    logger.info("Building TensorRT engine at %s", engine_path)
    onnx2engine(
        str(onnx_path),
        engine_path,
        workspace=workspace,
        half=half,
        int8=int8,
        dynamic=dynamic,
        shape=shape,
        metadata=metadata,
        verbose=verbose,
        prefix="TensorRT: ",
    )

    if not engine_path.is_file():
        raise FileNotFoundError(f"TensorRT export failed: {engine_path}")

    return str(onnx_path.resolve()), str(engine_path.resolve())
