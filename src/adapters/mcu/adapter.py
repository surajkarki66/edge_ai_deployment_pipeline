from __future__ import annotations

from typing import Any

from adapters.base import PlatformAdapter
from adapters.mcu.toolchains.registry import MCU_TOOLCHAINS
from core.registry import platform_registry
from core.types import DeploymentArtifact, PlatformConfig


@platform_registry.register("mcu")
class MCUAdapter(PlatformAdapter):
    platform_id = "mcu"

    def _resolve_toolchain(self, config: PlatformConfig) -> str:
        toolchain = str(config.get("toolchain", config.get("default_toolchain", "stm32n6")))
        if toolchain not in MCU_TOOLCHAINS:
            raise ValueError(
                f"Unknown MCU toolchain '{toolchain}'. "
                f"Choose from: {', '.join(MCU_TOOLCHAINS)}"
            )
        return toolchain

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        return MCU_TOOLCHAINS[self._resolve_toolchain(config)].validate(model, config)

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        return MCU_TOOLCHAINS[self._resolve_toolchain(config)].convert(model, config)

    def build_artifact(
        self, converted_model: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact:
        return MCU_TOOLCHAINS[self._resolve_toolchain(config)].build_artifact(
            converted_model, config
        )
