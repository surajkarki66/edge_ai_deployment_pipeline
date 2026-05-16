"""ARM edge adapter — ncnn."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adapters.base import PlatformAdapter
from core.registry import platform_registry
from core.types import DeploymentArtifact, PlatformConfig


@platform_registry.register("arm")
class ARMAdapter(PlatformAdapter):
    platform_id = "arm"

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        return str(config.get("toolchain", "ncnn")) in ("ncnn",)

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        toolchain = str(config.get("toolchain", "ncnn"))
        return {
            "param_bin": "weights",
            "bin_bin": "activations",
            "model_ref": model,
            "toolchain": toolchain,
        }

    def build_artifact(
        self, converted_model: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact:
        output_dir = Path(config.get("output_dir", "artifacts"))
        output_dir.mkdir(parents=True, exist_ok=True)
        platform_id = config.get("platform_id", self.platform_id)
        param_path = output_dir / f"{platform_id}.param"
        bin_path = output_dir / f"{platform_id}.bin"

        metadata = {
            "toolchain": converted_model["toolchain"],
            "precision": config.get("default_precision", "fp32"),
            "compile_flags": config.get("compile_flags", {}),
        }
        param_path.write_text("# ncnn param stub\n")
        bin_path.write_bytes(b"\x00\x00\x00\x00")

        return DeploymentArtifact(
            platform_id=platform_id,
            artifact_path=str(param_path.resolve()),
            toolchain_used=str(converted_model["toolchain"]),
            metadata={**metadata, "bin_path": str(bin_path.resolve())},
        )
