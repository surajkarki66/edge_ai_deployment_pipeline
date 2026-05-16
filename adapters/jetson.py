"""NVIDIA Jetson adapter — TensorRT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adapters.base import PlatformAdapter
from core.registry import platform_registry
from core.types import DeploymentArtifact, PlatformConfig


@platform_registry.register("jetson")
class JetsonAdapter(PlatformAdapter):
    platform_id = "jetson"

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        if str(config.get("toolchain", "tensorrt")) != "tensorrt":
            return False
        precision = config.get("default_precision", "fp16")
        return precision in ("fp32", "fp16", "int8")

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        return {
            "engine": "tensorrt_plan",
            "model_ref": model,
            "precision": config.get("default_precision", "fp16"),
            "toolchain": "tensorrt",
        }

    def build_artifact(
        self, converted_model: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact:
        output_dir = Path(config.get("output_dir", "artifacts"))
        output_dir.mkdir(parents=True, exist_ok=True)
        platform_id = config.get("platform_id", self.platform_id)
        artifact_path = output_dir / f"{platform_id}.engine"

        metadata = {
            "precision": converted_model["precision"],
            "toolchain": converted_model["toolchain"],
            "compile_flags": config.get("compile_flags", {}),
        }
        artifact_path.write_text(json.dumps({"stub": True, **metadata}, indent=2))

        return DeploymentArtifact(
            platform_id=platform_id,
            artifact_path=str(artifact_path.resolve()),
            toolchain_used=converted_model["toolchain"],
            metadata=metadata,
        )
