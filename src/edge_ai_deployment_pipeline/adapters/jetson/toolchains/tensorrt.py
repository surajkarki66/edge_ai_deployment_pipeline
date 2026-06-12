from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from edge_ai_deployment_pipeline.adapters.jetson.toolchains.base import (
    JetsonToolchain,
    SUPPORTED_PRECISIONS,
)
from edge_ai_deployment_pipeline.adapters.jetson.toolchains.yolo_export import export_yolo_to_tensorrt
from edge_ai_deployment_pipeline.adapters.jetson.toolchains.tensorrt_export import export_engine, export_engine_from_onnx
from edge_ai_deployment_pipeline.adapters.jetson.toolchains.tf2onnx_export import export_to_onnx, resolve_trt_shape_from_onnx
from edge_ai_deployment_pipeline.config.paths import DEFAULT_ARTIFACTS_DIR
from edge_ai_deployment_pipeline.config.task import resolve_task
from edge_ai_deployment_pipeline.core.types import (
    CvTask,
    DeploymentArtifact,
    FrameworkTag,
    PlatformConfig,
)
from edge_ai_deployment_pipeline.core.ultralytics_export import (
    yolo_calibration_yaml,
    yolo_export_int8,
)
from edge_ai_deployment_pipeline.core.yolo_loader import yolo_class_names

logger = logging.getLogger(__name__)


class TensorRTToolchain(JetsonToolchain):
    name = "tensorrt"

    def _is_ultralytics_object_detection(self, config: PlatformConfig) -> bool:
        return (
            resolve_task(config) == CvTask.OBJECT_DETECTION
            and config.get("framework") == FrameworkTag.ULTRALYTICS.value
        )

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        supported = config.get("supported_operations")
        if supported is not None and not supported:
            return False
        precision = str(config.get("default_precision", "fp16")).lower()
        if precision not in SUPPORTED_PRECISIONS:
            return False
        if not torch.cuda.is_available():
            logger.warning("TensorRT export requires CUDA; no GPU detected")
            return False

        if self._is_ultralytics_object_detection(config):
            if yolo_export_int8(config, default_precision=precision):
                if not yolo_calibration_yaml(config):
                    logger.warning(
                        "YOLO TensorRT INT8 requires yolo_export.calibration_data "
                        "(Ultralytics dataset yaml; not the image-folder calibration_data)"
                    )
                    return False
                try:
                    import tensorrt as trt

                    if int(trt.__version__.split(".", 1)[0]) >= 11:
                        logger.error(
                            "TensorRT INT8 calibration requires TensorRT 10.x "
                            "(pip install 'tensorrt>=10.16.1.11,<11'). Found %s.",
                            trt.__version__,
                        )
                        return False
                except ImportError:
                    logger.error("TensorRT is not installed")
                    return False
            return config.get("framework") == FrameworkTag.ULTRALYTICS.value

        if precision == "int8" and not config.get("calibration_data"):
            logger.warning("TensorRT INT8 requires calibration_data in platform config")
            return False
        if precision == "int8":
            try:
                import tensorrt as trt

                if int(trt.__version__.split(".", 1)[0]) >= 11:
                    logger.error(
                        "TensorRT INT8 calibration requires TensorRT 10.x "
                        "(pip install 'tensorrt>=10.16.1.11,<11'). Found %s.",
                        trt.__version__,
                    )
                    return False
            except ImportError:
                logger.error("TensorRT is not installed")
                return False

        framework = config.get("framework")
        if framework == FrameworkTag.PYTORCH.value:
            return self.is_pytorch_model(model)
        if framework == FrameworkTag.KERAS.value:
            if str(config.get("original_format", "")) in ("keras", "h5"):
                return self.is_keras_model(model)
            return False
        return False

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        channels, height, width = self._require_input_shape(config)
        precision = self.resolve_precision(config)

        if self._is_ultralytics_object_detection(config):
            source_path = config.get("source_path")
            if not source_path:
                raise ValueError("source_path is required for Ultralytics TensorRT export")
            return {
                "source_path": str(source_path),
                "input_shape": (channels, height, width),
                "precision": precision,
                "toolchain": self.name,
                "export_backend": "yolo_tensorrt",
                "class_names": yolo_class_names(model),
            }

        half, int8 = self.precision_to_trt_flags(precision)
        dynamic = bool(config.get("dynamic", False))
        opset = int(config.get("tf2onnx_opset", 13))
        calib_batch_size = int(config.get("calib_batch_size", 1))
        channels_last = bool(config.get("channels_last", True))
        calibration_data: str | None = None
        if int8:
            calibration_data = str(self._require_calibration_data(config))

        base: dict[str, Any] = {
            "precision": precision,
            "half": half,
            "int8": int8,
            "dynamic": dynamic,
            "input_shape": (channels, height, width),
            "toolchain": self.name,
            "tf2onnx_opset": opset,
            "calib_batch_size": calib_batch_size,
            "channels_last": channels_last,
            "calibration_data": calibration_data,
        }

        if self.is_tensorflow_export(model, config):
            self._require_tensorflow(config)
            base.update(
                {
                    "export_backend": "tf2onnx",
                    "source_path": str(config.get("source_path", "")),
                    "original_format": str(config.get("original_format", "")),
                    "model": model,
                }
            )
            return base

        self._require_pytorch(config)
        device = str(config.get("device", "cuda:0"))
        resolved = self.resolve_pytorch_model(model)
        example_input = torch.randn(1, channels, height, width, device=device)
        base.update(
            {
                "export_backend": "torch",
                "model": resolved,
                "example_input": example_input,
                "device": device,
            }
        )
        return base

    def build_artifact(
        self, converted: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact:
        platform_id = str(config.get("platform_id", config.get("board_id", "jetson")))
        board_dir = Path(config.get("output_dir", DEFAULT_ARTIFACTS_DIR)) / platform_id
        board_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "toolchain": self.name,
            "precision": converted["precision"],
            "input_shape": list(converted["input_shape"]),
            "compile_flags": config.get("compile_flags", {}),
            "dynamic": converted.get("dynamic", False),
            "export_backend": converted.get("export_backend", "torch"),
        }

        if converted.get("export_backend") == "yolo_tensorrt":
            _, height, width = converted["input_shape"]
            engine_path = export_yolo_to_tensorrt(
                source_path=Path(str(converted["source_path"])),
                output_dir=board_dir,
                height=height,
                width=width,
                config=config,
                precision=str(converted["precision"]),
            )
            yolo_metadata = {
                **metadata,
                "export_backend": "yolo_tensorrt",
                "engine_path": str(engine_path.resolve()),
                "class_names": converted.get("class_names", []),
                "task": CvTask.OBJECT_DETECTION.value,
            }
            return DeploymentArtifact(
                platform_id=platform_id,
                artifact_path=str(engine_path.resolve()),
                toolchain_used=self.name,
                metadata=yolo_metadata,
            )

        if converted.get("export_backend") == "tf2onnx":
            return self._build_from_tensorflow(converted, config, platform_id, board_dir, metadata)

        logger.info("Exporting PyTorch model to TensorRT in %s", board_dir)
        onnx_path, engine_path = export_engine(
            converted["model"],
            converted["example_input"],
            board_dir,
            half=converted["half"],
            int8=converted["int8"],
            dynamic=converted["dynamic"],
            metadata=metadata,
            verbose=bool(config.get("verbose", False)),
            device=converted["device"],
            calibration_data=converted.get("calibration_data"),
            channels_last=False,
            calib_batch_size=int(converted.get("calib_batch_size", 1)),
        )

        return DeploymentArtifact(
            platform_id=platform_id,
            artifact_path=engine_path,
            toolchain_used=self.name,
            metadata={**metadata, "onnx_path": onnx_path, "engine_path": engine_path},
        )

    def _build_from_tensorflow(
        self,
        converted: dict[str, Any],
        config: PlatformConfig,
        platform_id: str,
        board_dir: Path,
        metadata: dict[str, Any],
    ) -> DeploymentArtifact:
        channels, height, width = converted["input_shape"]
        onnx_path = board_dir / "model.onnx"

        logger.info("Converting Keras model to ONNX via tf2onnx")
        export_to_onnx(
            converted["model"],
            converted["source_path"],
            onnx_path,
            channels=channels,
            height=height,
            width=width,
            original_format=str(converted.get("original_format", "")),
            opset=int(converted.get("tf2onnx_opset", 13)),
            input_name=str(config.get("onnx_input_name", "input")),
            channels_last=bool(config.get("channels_last", True)),
        )

        channels_last = bool(config.get("channels_last", True))
        if channels_last:
            fallback = (1, height, width, channels)
        else:
            fallback = (1, channels, height, width)
        trt_shape = resolve_trt_shape_from_onnx(onnx_path, fallback=fallback)
        metadata["onnx_input_shape"] = list(trt_shape)

        logger.info("Building TensorRT engine from ONNX in %s", board_dir)
        onnx_str, engine_path = export_engine_from_onnx(
            onnx_path,
            board_dir,
            trt_shape=trt_shape,
            half=converted["half"],
            int8=converted["int8"],
            dynamic=converted["dynamic"],
            metadata=metadata,
            verbose=bool(config.get("verbose", False)),
            calibration_data=converted.get("calibration_data"),
            channels_last=bool(converted.get("channels_last", True)),
            calib_batch_size=int(converted.get("calib_batch_size", 1)),
        )

        return DeploymentArtifact(
            platform_id=platform_id,
            artifact_path=engine_path,
            toolchain_used=self.name,
            metadata={
                **metadata,
                "onnx_path": onnx_str,
                "engine_path": engine_path,
                "original_format": converted.get("original_format"),
            },
        )
