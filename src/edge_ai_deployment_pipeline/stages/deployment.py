from __future__ import annotations

import logging

from pathlib import Path

from edge_ai_deployment_pipeline.adapters.base import PlatformAdapter
from edge_ai_deployment_pipeline.config.catalogue import PlatformCatalogue
from edge_ai_deployment_pipeline.core.types import (
    DeploymentArtifact,
    LoadedModel,
    PlatformConfig,
    SelectedPlatform,
)

logger = logging.getLogger(__name__)


def _resolve_config_path(raw: str, config_path: str | Path | None) -> Path:
    """Resolve a config path; relative paths search cwd then ancestors of the catalogue file."""
    path = Path(raw)
    if path.is_absolute():
        return path

    candidates = [Path.cwd() / path]
    if config_path is not None:
        anchor = Path(config_path).resolve().parent
        for _ in range(8):
            candidates.append(anchor / path)
            anchor = anchor.parent

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _get_platform_adapter(adapter_id: str) -> PlatformAdapter:
    if adapter_id == "arm":
        from edge_ai_deployment_pipeline.adapters.arm.adapter import ARMAdapter

        return ARMAdapter()
    if adapter_id == "fpga":
        from edge_ai_deployment_pipeline.adapters.fpga.adapter import FPGAAdapter

        return FPGAAdapter()
    if adapter_id == "jetson":
        from edge_ai_deployment_pipeline.adapters.jetson.adapter import JetsonAdapter

        return JetsonAdapter()
    if adapter_id == "mcu":
        from edge_ai_deployment_pipeline.adapters.mcu.adapter import MCUAdapter

        return MCUAdapter()

    known = ("arm", "fpga", "jetson", "mcu")
    raise KeyError(
        f"Unknown platform adapter '{adapter_id}'. Choose from: {', '.join(known)}"
    )


class DeploymentStage:
    """Dispatch to the platform-family adapter for the selected board."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        loaded: LoadedModel,
        selected: SelectedPlatform,
        platform_config_path: str | Path | None = None,
        *,
        input_shape: tuple[int, int, int] | None = None,
    ) -> DeploymentArtifact:
        board_id = selected.platform_id
        config = self._load_board_config(
            board_id,
            platform_config_path,
            toolchain=selected.toolchain,
            loaded=loaded,
            input_shape=input_shape,
        )
        adapter_id = str(config.get("adapter_id", selected.platform_family))

        adapter = _get_platform_adapter(adapter_id)
        tc_note = f", toolchain={selected.toolchain}" if selected.toolchain else ""
        logger.info(
            "Deploying to %s (%s%s) via %s",
            board_id,
            selected.platform_family,
            tc_note,
            type(adapter).__name__,
        )

        if not adapter.validate(loaded.model, config):
            fmt = getattr(loaded, "original_format", "")
            hint = ""
            if fmt == "torchscript" and adapter_id == "fpga":
                hint = (
                    " Vitis-AI requires an eager PyTorch model (torch.save(nn.Module)), "
                    "not TorchScript — see toolchain error if raised during convert."
                )
            elif adapter_id == "jetson" and selected.toolchain == "tensorrt":
                try:
                    import torch

                    if not torch.cuda.is_available():
                        hint = (
                            " TensorRT engine build requires CUDA on this machine "
                            "(NVIDIA GPU with PyTorch CUDA). Build on a CUDA host or "
                            "on the target Jetson device."
                        )
                except ImportError:
                    pass
            raise ValueError(f"Adapter validation failed for board '{board_id}'.{hint}")

        converted = adapter.convert(loaded.model, config)
        artifact = adapter.build_artifact(converted, config)
        artifact.metadata["board_id"] = board_id
        artifact.metadata["platform_family"] = selected.platform_family
        if selected.toolchain:
            artifact.metadata["toolchain"] = selected.toolchain
        return artifact

    def _load_board_config(
        self,
        board_id: str,
        config_path: str | Path | None,
        *,
        toolchain: str | None = None,
        loaded: LoadedModel | None = None,
        input_shape: tuple[int, int, int] | None = None,
    ) -> PlatformConfig:
        catalogue = PlatformCatalogue.load(config_path)
        resolved_tc = toolchain or catalogue.list_toolchains(board_id)[0]
        spec = catalogue.resolve_board_config(board_id, resolved_tc)
        spec["output_dir"] = str(self.output_dir)
        spec["platform_id"] = board_id
        spec["board_id"] = board_id
        if input_shape is not None:
            spec["input_shape"] = input_shape
        if loaded is not None:
            spec["framework"] = loaded.framework.value
            spec["source_path"] = loaded.source_path
            spec["original_format"] = loaded.original_format
        if spec.get("calibration_data"):
            spec["calibration_data"] = str(
                _resolve_config_path(str(spec["calibration_data"]), config_path)
            )
        return spec
