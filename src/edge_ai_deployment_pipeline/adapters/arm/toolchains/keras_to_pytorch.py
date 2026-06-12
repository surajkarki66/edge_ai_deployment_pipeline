from __future__ import annotations

import logging

from pathlib import Path
from typing import Any
from onnx2torch import convert as onnx2torch_convert

from edge_ai_deployment_pipeline.adapters.jetson.toolchains.tf2onnx_export import export_to_onnx
from edge_ai_deployment_pipeline.config.paths import DEFAULT_ARTIFACTS_DIR
from edge_ai_deployment_pipeline.core.types import PlatformConfig

logger = logging.getLogger(__name__)


def keras_to_pytorch_module(
    model: Any,
    config: PlatformConfig,
    *,
    channels: int,
    height: int,
    width: int,
) -> tuple[Any, Path]:
    """Convert a Keras model to PyTorch via ONNX."""
    platform_id = str(config.get("platform_id", config.get("board_id", "arm")))
    board_dir = Path(config.get("output_dir", DEFAULT_ARTIFACTS_DIR)) / platform_id
    board_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = board_dir / "model.onnx"

    export_to_onnx(
        model,
        str(config.get("source_path", "")),
        onnx_path,
        channels=channels,
        height=height,
        width=width,
        original_format=str(config.get("original_format", "")),
        opset=int(config.get("tf2onnx_opset", 13)),
        input_name=str(config.get("onnx_input_name", "input")),
        channels_last=bool(config.get("channels_last", True)),
    )
    logger.info("Converting ONNX to PyTorch via onnx2torch")
    torch_model = onnx2torch_convert(str(onnx_path))
    return torch_model.eval(), onnx_path.resolve()
