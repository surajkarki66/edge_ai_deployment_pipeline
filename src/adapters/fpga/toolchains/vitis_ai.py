"""AMD Vitis AI — DPUCZDX8G on Zynq UltraScale+ MPSoC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adapters.fpga.toolchains.base import FpgaToolchain
from core.types import DeploymentArtifact, PlatformConfig

DPU_ARCH = "DPUCZDX8G"


class VitisAiToolchain(FpgaToolchain):
    name = "vitis_ai"

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        if not config.get("supported_operations"):
            return False
        return str(config.get("dpu_arch", DPU_ARCH)) == DPU_ARCH

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        return {
            "format": "xmodel",
            "model_ref": model,
            "toolchain": self.name,
            "dpu_arch": DPU_ARCH,
            "dpu_silicon_family": config.get(
                "dpu_silicon_family", "Zynq UltraScale+ MPSoC"
            ),
            "quantization": config.get("quantization", "int8"),
        }

    def build_artifact(self, converted: dict[str, Any], config: PlatformConfig) -> DeploymentArtifact:
        output_dir = Path(config.get("output_dir", "artifacts"))
        output_dir.mkdir(parents=True, exist_ok=True)
        board_id = config.get("board_id", "fpga")
        path = output_dir / f"{board_id}_vitis_ai.xmodel"
        metadata = {
            "toolchain": self.name,
            "dpu_arch": converted["dpu_arch"],
            "dpu_silicon_family": converted.get("dpu_silicon_family"),
            "quantization": converted["quantization"],
        }
        path.write_text(json.dumps({"stub": True, **metadata}, indent=2))
        return DeploymentArtifact(
            platform_id=board_id,
            artifact_path=str(path.resolve()),
            toolchain_used=self.name,
            metadata=metadata,
        )
