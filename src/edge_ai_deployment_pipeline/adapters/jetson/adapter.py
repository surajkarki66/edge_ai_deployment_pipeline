from __future__ import annotations

from typing import Any

from edge_ai_deployment_pipeline.adapters.base import PlatformAdapter
from edge_ai_deployment_pipeline.adapters.jetson.toolchains.registry import JETSON_TOOLCHAINS
from edge_ai_deployment_pipeline.core.types import DeploymentArtifact, PlatformConfig


class JetsonAdapter(PlatformAdapter):
    platform_id = "jetson"

    def _resolve_toolchain(self, config: PlatformConfig) -> str:
        toolchain = str(config.get("toolchain", config.get("default_toolchain", "tensorrt")))
        if toolchain not in JETSON_TOOLCHAINS:
            raise ValueError(
                f"Unknown Jetson toolchain '{toolchain}'. "
                f"Choose from: {', '.join(JETSON_TOOLCHAINS)}"
            )
        return toolchain

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        return JETSON_TOOLCHAINS[self._resolve_toolchain(config)].validate(model, config)

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        return JETSON_TOOLCHAINS[self._resolve_toolchain(config)].convert(model, config)

    def build_artifact(
        self, converted_model: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact:
        return JETSON_TOOLCHAINS[self._resolve_toolchain(config)].build_artifact(
            converted_model, config
        )
