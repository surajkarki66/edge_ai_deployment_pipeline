"""Base contract for a Jetson deployment toolchain."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.types import DeploymentArtifact, FrameworkTag, PlatformConfig

SUPPORTED_PRECISIONS = frozenset({"fp32", "fp16", "int8"})


class JetsonToolchain(ABC):
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
    def resolve_precision(config: PlatformConfig, *, default: str = "fp16") -> str:
        precision = str(config.get("default_precision", default)).lower()
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"Unsupported precision '{precision}' for {config.get('toolchain', 'jetson')}. "
                f"Choose from: {', '.join(sorted(SUPPORTED_PRECISIONS))}"
            )
        return precision

    @staticmethod
    def precision_to_trt_flags(precision: str) -> tuple[bool, bool]:
        """Map ``default_precision`` to TensorRT ``half`` / ``int8`` flags."""
        if precision == "fp16":
            return True, False
        if precision == "int8":
            return False, True
        return False, False

    @staticmethod
    def resolve_workspace_gb(config: PlatformConfig) -> int | None:
        compile_flags = config.get("compile_flags", {})
        if not isinstance(compile_flags, dict):
            return None
        mb = compile_flags.get("workspace_size_mb")
        if mb is None:
            return None
        return max(1, int(int(mb) / 1024)) or 1

    def _require_input_shape(self, config: PlatformConfig) -> tuple[int, int, int]:
        input_shape = config.get("input_shape")
        if not input_shape or len(input_shape) != 3:
            raise ValueError(
                f"input_shape (channels, height, width) is required for {self.name} export"
            )
        channels, height, width = (int(d) for d in input_shape)
        return channels, height, width

    @staticmethod
    def is_pytorch_model(model: Any) -> bool:
        try:
            import torch.nn as nn
        except ImportError:
            return False
        return isinstance(model, nn.Module) or hasattr(model, "forward")

    @staticmethod
    def resolve_pytorch_model(model: Any) -> Any:
        import torch.nn as nn

        if isinstance(model, nn.Module):
            return model.eval()
        if hasattr(model, "forward"):
            return model.eval()
        raise TypeError(
            f"Jetson export requires a PyTorch nn.Module or TorchScript model, got {type(model).__name__}"
        )
