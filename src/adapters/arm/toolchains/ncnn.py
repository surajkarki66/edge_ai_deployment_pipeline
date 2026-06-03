from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

from adapters.arm.toolchains.base import ArmToolchain
from adapters.arm.toolchains.ncnn_export import torch2ncnn
from core.types import DeploymentArtifact, FrameworkTag, PlatformConfig

logger = logging.getLogger(__name__)


class NcnnToolchain(ArmToolchain):
    name = "ncnn"

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        if config.get("framework") != FrameworkTag.PYTORCH.value:
            return False
        return self.is_pytorch_model(model)

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        import torch

        self._require_pytorch(config)
        channels, height, width = self._require_input_shape(config)
        resolved = self.resolve_pytorch_model(model)
        example_input = torch.randn(1, channels, height, width)
        fp16 = self.resolve_fp16(config)

        return {
            "model": resolved,
            "example_input": example_input,
            "half": fp16,
            "toolchain": self.name,
        }

    def build_artifact(
        self, converted: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact:
        platform_id = str(config.get("platform_id", config.get("board_id", "arm")))
        board_dir = Path(config.get("output_dir", "artifacts")) / platform_id
        board_dir.mkdir(parents=True, exist_ok=True)

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
        }

        return DeploymentArtifact(
            platform_id=platform_id,
            artifact_path=str(param_path.resolve()),
            toolchain_used=self.name,
            metadata=metadata,
        )
