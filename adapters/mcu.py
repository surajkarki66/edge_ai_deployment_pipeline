"""MCU adapter — STM32CubeAI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adapters.base import PlatformAdapter
from core.registry import platform_registry
from core.types import DeploymentArtifact, PlatformConfig


@platform_registry.register("mcu")
class MCUAdapter(PlatformAdapter):
    platform_id = "mcu"

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        return True  # Full validation deferred to CubeAI

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        toolchain = str(config.get("toolchain", "stm32cubeai"))
        return {
            "c_model": "generated_weights",
            "model_ref": model,
            "toolchain": toolchain,
            "target_mcu": config.get("target_mcu", "STM32H747"),
        }

    def build_artifact(
        self, converted_model: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact:
        output_dir = Path(config.get("output_dir", "artifacts"))
        output_dir.mkdir(parents=True, exist_ok=True)
        platform_id = config.get("platform_id", self.platform_id)
        artifact_path = output_dir / f"{platform_id}.c"

        metadata = {
            "toolchain": converted_model["toolchain"],
            "target_mcu": converted_model["target_mcu"],
            "quantization": config.get("quantization", "int8"),
        }
        artifact_path.write_text(
            f"/* STM32CubeAI stub for {platform_id} */\n"
            + json.dumps({"stub": True, **metadata}, indent=2)
        )

        return DeploymentArtifact(
            platform_id=platform_id,
            artifact_path=str(artifact_path.resolve()),
            toolchain_used=str(converted_model["toolchain"]),
            metadata=metadata,
        )
