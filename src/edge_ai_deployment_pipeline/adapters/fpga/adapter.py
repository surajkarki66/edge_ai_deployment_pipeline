from __future__ import annotations

from typing import Any

from edge_ai_deployment_pipeline.adapters.base import PlatformAdapter
from edge_ai_deployment_pipeline.adapters.fpga.toolchains.registry import FPGA_TOOLCHAINS
from edge_ai_deployment_pipeline.core.types import DeploymentArtifact, PlatformConfig


class FPGAAdapter(PlatformAdapter):
    platform_id = "fpga"

    def _resolve_toolchain(self, config: PlatformConfig) -> str:
        toolchain = str(config.get("toolchain", config.get("default_toolchain", "vitis_ai")))
        if toolchain not in FPGA_TOOLCHAINS:
            raise ValueError(
                f"Unknown FPGA toolchain '{toolchain}'. "
                f"Choose from: {', '.join(FPGA_TOOLCHAINS)}"
            )
        return toolchain

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        return FPGA_TOOLCHAINS[self._resolve_toolchain(config)].validate(model, config)

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        return FPGA_TOOLCHAINS[self._resolve_toolchain(config)].convert(model, config)

    def build_artifact(self, converted_model: dict[str, Any], config: PlatformConfig) -> DeploymentArtifact:
        return FPGA_TOOLCHAINS[self._resolve_toolchain(config)].build_artifact(
            converted_model, config
        )
