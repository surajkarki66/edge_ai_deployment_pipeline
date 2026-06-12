from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

from edge_ai_deployment_pipeline.adapters.arm.toolchains.base import ArmToolchain
from edge_ai_deployment_pipeline.config.paths import DEFAULT_ARTIFACTS_DIR
from edge_ai_deployment_pipeline.adapters.arm.toolchains.keras_to_pytorch import keras_to_pytorch_module
from edge_ai_deployment_pipeline.adapters.arm.toolchains.ncnn_export import torch2ncnn
from edge_ai_deployment_pipeline.adapters.arm.toolchains.yolo_export import export_yolo_to_ncnn
from edge_ai_deployment_pipeline.config.task import resolve_task
from edge_ai_deployment_pipeline.core.types import (
    CvTask,
    DeploymentArtifact,
    FrameworkTag,
    PlatformConfig,
)
from edge_ai_deployment_pipeline.core.yolo_loader import yolo_class_names

logger = logging.getLogger(__name__)


class NcnnToolchain(ArmToolchain):
    name = "ncnn"

    def _is_ultralytics_object_detection(self, config: PlatformConfig) -> bool:
        return (
            resolve_task(config) == CvTask.OBJECT_DETECTION
            and config.get("framework") == FrameworkTag.ULTRALYTICS.value
        )

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        if self._is_ultralytics_object_detection(config):
            return config.get("framework") == FrameworkTag.ULTRALYTICS.value

        framework = config.get("framework")
        if framework == FrameworkTag.PYTORCH.value:
            return self.is_pytorch_model(model)
        if framework == FrameworkTag.KERAS.value:
            return self.is_keras_format(config) and self.is_keras_model(model)
        return False

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        channels, height, width = self._require_input_shape(config)

        if self._is_ultralytics_object_detection(config):
            source_path = config.get("source_path")
            if not source_path:
                raise ValueError("source_path is required for YOLO NCNN export")
            export_cfg = config.get("yolo_export")
            half = bool(export_cfg.get("half", False)) if isinstance(export_cfg, dict) else False
            return {
                "source_path": str(source_path),
                "input_shape": (channels, height, width),
                "half": half,
                "toolchain": self.name,
                "export_backend": "yolo_ncnn",
                "class_names": yolo_class_names(model),
            }

        fp16 = self.resolve_fp16(config)
        framework = config.get("framework")
        channels_last = bool(config.get("channels_last", True))
        export_backend = "pytorch"
        onnx_path: str | None = None

        if framework == FrameworkTag.PYTORCH.value:
            resolved = self.resolve_pytorch_model(model)
            example_input = self.example_input_tensor(
                channels, height, width, channels_last=False
            )
        elif framework == FrameworkTag.KERAS.value:
            self._require_keras(config)
            resolved, onnx_file = keras_to_pytorch_module(
                model,
                config,
                channels=channels,
                height=height,
                width=width,
            )
            export_backend = "keras_onnx2torch"
            onnx_path = str(onnx_file)
            example_input = self.example_input_tensor(
                channels, height, width, channels_last=channels_last
            )
        else:
            raise ValueError(
                f"{self.name} toolchain requires a PyTorch or Keras model, got {framework!r}"
            )

        return {
            "model": resolved,
            "example_input": example_input,
            "half": fp16,
            "toolchain": self.name,
            "export_backend": export_backend,
            "onnx_path": onnx_path,
        }

    def build_artifact(
        self, converted: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact:
        platform_id = str(config.get("platform_id", config.get("board_id", "arm")))
        board_dir = Path(config.get("output_dir", DEFAULT_ARTIFACTS_DIR)) / platform_id
        board_dir.mkdir(parents=True, exist_ok=True)

        if converted.get("export_backend") == "yolo_ncnn":
            _, height, width = converted["input_shape"]
            param_path, bin_path, ncnn_py, ncnn_model_dir = export_yolo_to_ncnn(
                source_path=Path(str(converted["source_path"])),
                output_dir=board_dir,
                height=height,
                width=width,
                config=config,
            )
            metadata = {
                "toolchain": self.name,
                "precision": "fp16" if converted["half"] else "fp32",
                "compile_flags": config.get("compile_flags", {}),
                "bin_path": str(bin_path.resolve()),
                "ncnn_model_dir": str(ncnn_model_dir.resolve()),
                "export_backend": "yolo_ncnn",
                "class_names": converted.get("class_names", []),
                "task": CvTask.OBJECT_DETECTION.value,
            }
            if ncnn_py is not None:
                metadata["ncnn_py"] = str(ncnn_py.resolve())
            return DeploymentArtifact(
                platform_id=platform_id,
                artifact_path=str(param_path.resolve()),
                toolchain_used=self.name,
                metadata=metadata,
            )

        logger.info("Exporting PyTorch model to NCNN via PNNX in %s", board_dir)
        torch2ncnn(
            converted["model"],
            converted["example_input"],
            board_dir,
            half=converted["half"],
            device="cpu",
        )

        param_path = board_dir / "model.ncnn.param"
        bin_path = board_dir / "model.ncnn.bin"
        if not param_path.is_file() or not bin_path.is_file():
            raise FileNotFoundError(
                f"NCNN export did not produce expected files in {board_dir}"
            )

        metadata = {
            "toolchain": self.name,
            "precision": "fp16" if converted["half"] else "fp32",
            "compile_flags": config.get("compile_flags", {}),
            "bin_path": str(bin_path.resolve()),
            "ncnn_py": str((board_dir / "model_ncnn.py").resolve()),
            "export_backend": converted.get("export_backend", "pytorch"),
        }
        if converted.get("onnx_path"):
            metadata["onnx_path"] = converted["onnx_path"]

        return DeploymentArtifact(
            platform_id=platform_id,
            artifact_path=str(param_path.resolve()),
            toolchain_used=self.name,
            metadata=metadata,
        )
