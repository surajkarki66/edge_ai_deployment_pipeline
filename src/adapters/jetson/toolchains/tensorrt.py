"""NVIDIA TensorRT deployment — PyTorch → ONNX → engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from adapters.jetson.toolchains.base import JetsonToolchain, SUPPORTED_PRECISIONS
from adapters.jetson.toolchains.tensorrt_export import export_engine
from core.types import DeploymentArtifact, FrameworkTag, PlatformConfig

logger = logging.getLogger(__name__)


class TensorRTToolchain(JetsonToolchain):
    name = "tensorrt"

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        if config.get("framework") not in (None, FrameworkTag.PYTORCH.value):
            return False
        precision = str(config.get("default_precision", "fp16")).lower()
        if precision not in SUPPORTED_PRECISIONS:
            return False
        if not torch.cuda.is_available():
            logger.warning("TensorRT export requires CUDA; no GPU detected")
            return False
        return self.is_pytorch_model(model)

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        self._require_pytorch(config)
        channels, height, width = self._require_input_shape(config)
        resolved = self.resolve_pytorch_model(model)
        precision = self.resolve_precision(config)
        half, int8 = self.precision_to_trt_flags(precision)

        device = str(config.get("device", "cuda:0"))
        example_input = torch.randn(1, channels, height, width, device=device)

        return {
            "model": resolved,
            "example_input": example_input,
            "precision": precision,
            "half": half,
            "int8": int8,
            "dynamic": bool(config.get("dynamic", False)),
            "workspace": self.resolve_workspace_gb(config),
            "input_shape": (channels, height, width),
            "toolchain": self.name,
            "device": device,
        }

    def build_artifact(
        self, converted: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact:
        platform_id = str(config.get("platform_id", config.get("board_id", "jetson")))
        board_dir = Path(config.get("output_dir", "artifacts")) / platform_id
        board_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "toolchain": self.name,
            "precision": converted["precision"],
            "input_shape": list(converted["input_shape"]),
            "compile_flags": config.get("compile_flags", {}),
            "dynamic": converted["dynamic"],
        }

        logger.info("Exporting PyTorch model to TensorRT in %s", board_dir)
        onnx_path, engine_path = export_engine(
            converted["model"],
            converted["example_input"],
            board_dir,
            workspace=converted["workspace"],
            half=converted["half"],
            int8=converted["int8"],
            dynamic=converted["dynamic"],
            metadata=metadata,
            verbose=bool(config.get("verbose", False)),
            device=converted["device"],
        )

        return DeploymentArtifact(
            platform_id=platform_id,
            artifact_path=engine_path,
            toolchain_used=self.name,
            metadata={
                **metadata,
                "onnx_path": onnx_path,
                "engine_path": engine_path,
            },
        )
