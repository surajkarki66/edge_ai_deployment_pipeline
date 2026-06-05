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
        if config.get("framework") != FrameworkTag.PYTORCH.value:
            raise ValueError(f"{self.name} toolchain requires a PyTorch model")

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
            import torch
            import torch.nn as nn
        except ImportError:
            return False
        if isinstance(model, torch.jit.ScriptModule):
            return False
        return isinstance(model, nn.Module) or hasattr(model, "forward")

    @staticmethod
    def require_eager_pytorch(
        *,
        original_format: str = "",
        model: Any | None = None,
    ) -> None:
        """Vitis-AI NNDCT quantizer does not support TorchScript/traced models."""
        import torch

        if original_format == "torchscript":
            raise ValueError(
                "Vitis-AI requires an eager PyTorch model (nn.Module), not TorchScript. "
                "Your file was loaded as torchscript. Re-save a full model, e.g.\n"
                "  import torchvision.models as models\n"
                "  m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)\n"
                "  torch.save(m, 'resnet18_eager.pt')\n"
                "Then run the pipeline with resnet18_eager.pt."
            )
        if model is not None and isinstance(model, torch.jit.ScriptModule):
            raise ValueError(
                "Vitis-AI requires an eager PyTorch nn.Module; got TorchScript. "
                "Use torch.save(model, path) on a regular Module, not torch.jit.save/trace."
            )

    @staticmethod
    def resolve_pytorch_model(model: Any) -> Any:
        import torch.nn as nn

        if isinstance(model, nn.Module):
            return model.eval()
        if hasattr(model, "forward"):
            return model.eval()
        raise TypeError(
            f"Vitis-AI export requires nn.Module or TorchScript, got {type(model).__name__}"
        )
