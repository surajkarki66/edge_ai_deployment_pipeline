"""ARM adapter — dispatches to per-toolchain backends (NCNN, …)."""

from __future__ import annotations

from typing import Any

from adapters.arm.toolchains.registry import ARM_TOOLCHAINS
from adapters.base import PlatformAdapter
from core.registry import platform_registry
from core.types import DeploymentArtifact, PlatformConfig


@platform_registry.register("arm")
class ARMAdapter(PlatformAdapter):
    platform_id = "arm"

    def _resolve_toolchain(self, config: PlatformConfig) -> str:
        toolchain = str(config.get("toolchain", config.get("default_toolchain", "ncnn")))
        if toolchain not in ARM_TOOLCHAINS:
            raise ValueError(
                f"Unknown ARM toolchain '{toolchain}'. "
                f"Choose from: {', '.join(ARM_TOOLCHAINS)}"
            )
        return toolchain

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        return ARM_TOOLCHAINS[self._resolve_toolchain(config)].validate(model, config)

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        return ARM_TOOLCHAINS[self._resolve_toolchain(config)].convert(model, config)

    def build_artifact(
        self, converted_model: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact:
        return ARM_TOOLCHAINS[self._resolve_toolchain(config)].build_artifact(
            converted_model, config
        )
