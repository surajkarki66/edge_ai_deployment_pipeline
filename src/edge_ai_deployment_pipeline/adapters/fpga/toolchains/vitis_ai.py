from __future__ import annotations

import logging
import shutil

from pathlib import Path
from typing import Any

from edge_ai_deployment_pipeline.adapters.fpga.toolchains.base import FpgaToolchain
from edge_ai_deployment_pipeline.adapters.fpga.toolchains.vitis_ai_docker import (
    _docker_available,
    export_pytorch_to_dpu,
    resolve_dpu_target,
)
from edge_ai_deployment_pipeline.core.types import DeploymentArtifact, FrameworkTag, PlatformConfig

logger = logging.getLogger(__name__)

DPU_ARCH = "DPUCZDX8G"


class VitisAiToolchain(FpgaToolchain):
    """PyTorch post-training quantization + DPU compilation via Vitis-AI Docker."""

    name = "vitis_ai"

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        if not config.get("supported_operations"):
            return False
        if config.get("framework") not in (None, FrameworkTag.PYTORCH.value):
            return False
        if not _docker_available():
            logger.warning("Vitis-AI export requires Docker")
            return False
        if not self.is_pytorch_model(model):
            return False
        fmt = str(config.get("original_format", ""))
        if fmt == "torchscript":
            logger.warning("Vitis-AI does not support TorchScript models")
            return False
        return True

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        self._require_pytorch(config)
        self.require_eager_pytorch(
            original_format=str(config.get("original_format", "")),
            model=model,
        )
        channels, height, width = self._require_input_shape(config)
        calibration_data = self._require_calibration_data(config)
        dpu_target = resolve_dpu_target(config)
        source_path = config.get("source_path")
        if not source_path:
            raise ValueError("source_path is required for Vitis-AI export")

        return {
            "source_path": str(source_path),
            "original_format": str(config.get("original_format", "")),
            "input_shape": (channels, height, width),
            "calibration_data": str(calibration_data),
            "dpu_target": dpu_target,
            "dpu_arch": str(config.get("dpu_arch", DPU_ARCH)),
            "dpu_silicon_family": config.get(
                "dpu_silicon_family", "Zynq UltraScale+ MPSoC"
            ),
            "quantization": config.get("quantization", "int8"),
            "toolchain": self.name,
        }

    def build_artifact(
        self, converted: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact:
        self.require_eager_pytorch(
            original_format=str(converted.get("original_format", "")),
        )
        platform_id = str(config.get("platform_id", config.get("board_id", "fpga")))
        board_dir = Path(config.get("output_dir", "artifacts")) / platform_id
        workspace = board_dir / "vitis_workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        channels, height, width = converted["input_shape"]
        model_path = self._stage_model_in_workspace(workspace, converted, config)
        logger.info("Staged model at %s for Vitis-AI workspace", model_path)

        calibration_data = Path(converted["calibration_data"])
        xmodel_path, compiled_dir = export_pytorch_to_dpu(
            workspace,
            model_path,
            calibration_data,
            channels=channels,
            height=height,
            width=width,
            dpu_target=converted["dpu_target"],
            config=config,
        )

        final_xmodel = board_dir / xmodel_path.name
        shutil.copy2(xmodel_path, final_xmodel)

        metadata = {
            "toolchain": self.name,
            "dpu_arch": converted["dpu_arch"],
            "dpu_target": converted["dpu_target"],
            "dpu_silicon_family": converted.get("dpu_silicon_family"),
            "quantization": converted["quantization"],
            "workspace": str(workspace.resolve()),
            "compiled_dir": str(compiled_dir.resolve()),
            "xmodel_path": str(final_xmodel.resolve()),
        }

        return DeploymentArtifact(
            platform_id=platform_id,
            artifact_path=str(final_xmodel.resolve()),
            toolchain_used=self.name,
            metadata=metadata,
        )

    def _stage_model_in_workspace(
        self,
        workspace: Path,
        converted: dict[str, Any],
        config: PlatformConfig,
    ) -> Path:
        """Copy the intake model file into the workspace; Docker loads it there."""
        model_path = workspace / "model.pt"
        source = Path(str(converted.get("source_path", config.get("source_path", ""))))
        if not source.is_file():
            raise FileNotFoundError(f"Model file not found: {source}")
        shutil.copy2(source, model_path)
        return model_path
