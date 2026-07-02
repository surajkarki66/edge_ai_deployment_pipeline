from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from edge_ai_deployment_pipeline.core.types import DeploymentArtifact, FrameworkTag, PlatformConfig


class McuToolchain(ABC):
    name: str = ""

    @abstractmethod
    def validate(self, model: Any, config: PlatformConfig) -> bool: ...

    @abstractmethod
    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]: ...

    @abstractmethod
    def build_artifact(
        self, converted: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact: ...

    def _require_keras(self, config: PlatformConfig) -> None:
        if config.get("framework") != FrameworkTag.KERAS.value:
            raise ValueError(f"{self.name} toolchain requires a Keras model (.h5 / .keras)")

    def _require_input_shape(self, config: PlatformConfig) -> tuple[int, int, int]:
        input_shape = config.get("input_shape")
        if not input_shape or len(input_shape) != 3:
            raise ValueError(
                f"input_shape (channels, height, width) is required for {self.name}"
            )
        return (int(input_shape[0]), int(input_shape[1]), int(input_shape[2]))

    def _require_calibration_data(self, config: PlatformConfig) -> Path:
        raw = config.get("calibration_data")
        if not raw:
            raise ValueError(
                "calibration_data is required for STM32N6 Keras PTQ "
                "(directory with class subfolders or images). "
                "YOLO object detection uses yolo_export.calibration_data instead."
            )
        path = Path(str(raw))
        if not path.is_dir():
            raise FileNotFoundError(f"calibration_data must be a directory: {path}")
        return path.resolve()

    def _resolve_c_project(self, config: PlatformConfig) -> Path:
        from edge_ai_deployment_pipeline.adapters.mcu.toolchains.stm32n6.c_projects import resolve_c_project

        return resolve_c_project(config)

    @staticmethod
    def is_keras_model(model: Any) -> bool:
        try:
            import tensorflow as tf
        except ImportError:
            return False
        return isinstance(model, tf.keras.Model)

    @staticmethod
    def infer_class_names(calib_dir: Path) -> list[str]:
        subdirs = sorted(p.name for p in calib_dir.iterdir() if p.is_dir())
        if subdirs:
            return subdirs
        return ["default"]
