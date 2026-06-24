from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from edge_ai_deployment_pipeline.core.types import DeploymentArtifact, FrameworkTag, PlatformConfig


class FpgaToolchain(ABC):
    name: str = ""

    @abstractmethod
    def validate(self, model: Any, config: PlatformConfig) -> bool: ...

    @abstractmethod
    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]: ...

    @abstractmethod
    def build_artifact(
        self, converted: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact: ...

    def _require_pytorch(self, config: PlatformConfig) -> None:
        if config.get("framework") not in (FrameworkTag.PYTORCH.value, FrameworkTag.ULTRALYTICS.value):
            raise ValueError(f"{self.name} toolchain requires a PyTorch model (.pt)")

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
                "calibration_data is required for Vitis-AI PTQ "
                "(path to val_ids.txt or an image directory)"
            )
        path = Path(str(raw))
        if not path.exists():
            raise FileNotFoundError(f"calibration_data not found: {path}")
        return path.resolve()

    @staticmethod
    def is_pytorch_model(model: Any) -> bool:
        try:
            import torch.nn as nn
        except ImportError:
            return False
        return isinstance(model, nn.Module)

    @staticmethod
    def resolve_pytorch_model(model: Any) -> Any:
        import torch.nn as nn

        if isinstance(model, nn.Module):
            return model.eval()
        raise TypeError(
            f"Vitis-AI export requires torch.nn.Module, got {type(model).__name__}"
        )
