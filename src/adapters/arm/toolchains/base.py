"""Base contract for an ARM deployment toolchain."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.types import DeploymentArtifact, FrameworkTag, PlatformConfig


class ArmToolchain(ABC):
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
        if config.get("framework") != FrameworkTag.PYTORCH.value:
            raise ValueError(f"{self.name} toolchain requires a PyTorch model")

    @staticmethod
    def resolve_fp16(config: PlatformConfig) -> bool:
        """NCNN/PNNX fp16 export flag (boolean, not a precision string)."""
        if "fp16" in config:
            return bool(config["fp16"])
        if "half" in config:
            return bool(config["half"])
        return False

    def _require_input_shape(self, config: PlatformConfig) -> tuple[int, int, int]:
        input_shape = config.get("input_shape")
        if not input_shape or len(input_shape) != 3:
            raise ValueError(
                f"input_shape (channels, height, width) is required for {self.name} export"
            )
        channels, height, width = (int(d) for d in input_shape)
        return channels, height, width

    @staticmethod
    def resolve_pytorch_model(model: Any) -> Any:
        import torch.nn as nn

        if isinstance(model, nn.Module):
            return model.eval()
        if hasattr(model, "forward"):
            return model.eval()
        raise TypeError(
            f"ARM export requires a PyTorch nn.Module or TorchScript model, got {type(model).__name__}"
        )

    @staticmethod
    def is_pytorch_model(model: Any) -> bool:
        try:
            import torch.nn as nn
        except ImportError:
            return False
        return isinstance(model, nn.Module) or hasattr(model, "forward")
