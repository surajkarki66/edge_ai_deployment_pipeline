from __future__ import annotations

import json
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
from edge_ai_deployment_pipeline.core.pytorch_loader import (
    resolve_model_definition_file,
    split_model_class_spec,
)
from edge_ai_deployment_pipeline.core.types import DeploymentArtifact, FrameworkTag, PlatformConfig

logger = logging.getLogger(__name__)

DPU_ARCH = "DPUCZDX8G"
WORKSPACE_MODEL_CLASS = "models/model_arch.py"
WORKSPACE_WEIGHTS = "model.pt"
WORKSPACE_META = "model_meta.json"


class VitisAiToolchain(FpgaToolchain):
    """PyTorch PTQ + DPU compilation via Vitis-AI Docker."""

    name = "vitis_ai"

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        if not config.get("supported_operations"):
            return False
        if not _docker_available():
            logger.warning("Vitis-AI export requires Docker")
            return False
        if config.get("framework") != FrameworkTag.PYTORCH.value:
            return False
        return self.is_pytorch_model(model)

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        channels, height, width = self._require_input_shape(config)
        calibration_data = self._require_calibration_data(config)
        dpu_target = resolve_dpu_target(config)
        source_path = config.get("source_path")
        if not source_path:
            raise ValueError("source_path is required for Vitis-AI export")
        architecture_spec = config.get("architecture_spec")
        if not architecture_spec:
            raise ValueError("FPGA export requires --model-class")

        self._require_pytorch(config)
        resolved = self.resolve_pytorch_model(model)

        return {
            "framework": FrameworkTag.PYTORCH.value,
            "source_path": str(source_path),
            "original_format": str(config.get("original_format", "")),
            "architecture_spec": str(architecture_spec),
            "model_kwargs": config.get("model_kwargs") or {},
            "input_shape": (channels, height, width),
            "calibration_data": str(calibration_data),
            "dpu_target": dpu_target,
            "dpu_arch": str(config.get("dpu_arch", DPU_ARCH)),
            "dpu_silicon_family": config.get(
                "dpu_silicon_family", "Zynq UltraScale+ MPSoC"
            ),
            "quantization": config.get("quantization", "int8"),
            "toolchain": self.name,
            "export_backend": "vitis_ai_pytorch",
            "model": resolved,
        }

    def build_artifact(
        self, converted: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact:
        platform_id = str(config.get("platform_id", config.get("board_id", "fpga")))
        board_dir = Path(config.get("output_dir", "artifacts")) / platform_id
        workspace = (board_dir / "vitis_workspace").resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        channels, height, width = converted["input_shape"]
        self._stage_model_in_workspace(workspace, converted)
        logger.info("Staged weights and model definition under %s", workspace)

        calibration_data = Path(converted["calibration_data"])
        xmodel_path, compiled_dir = export_pytorch_to_dpu(
            workspace.resolve(),
            Path(WORKSPACE_WEIGHTS),
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
            "export_backend": converted.get("export_backend", "vitis_ai_pytorch"),
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
    ) -> None:
        """Copy weights + architecture ``.py`` for Vitis-AI Docker (state_dict loading)."""
        source = Path(str(converted.get("source_path", "")))
        if not source.is_file():
            raise FileNotFoundError(f"Weights file not found: {source}")

        architecture_spec = str(converted.get("architecture_spec", ""))
        if not architecture_spec:
            raise ValueError("FPGA export requires architecture_spec from --model-class")

        _, class_name = split_model_class_spec(architecture_spec)
        definition_file = resolve_model_definition_file(architecture_spec)

        models_dir = workspace / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(definition_file, models_dir / "model_arch.py")
        shutil.copy2(source, workspace / WORKSPACE_WEIGHTS)

        meta = {
            "model_class": f"{WORKSPACE_MODEL_CLASS}:{class_name}",
            "model_kwargs": converted.get("model_kwargs") or {},
        }
        (workspace / WORKSPACE_META).write_text(json.dumps(meta) + "\n")
