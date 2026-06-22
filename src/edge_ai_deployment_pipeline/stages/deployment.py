from __future__ import annotations

import logging

from pathlib import Path

from edge_ai_deployment_pipeline.adapters.base import PlatformAdapter
from edge_ai_deployment_pipeline.config.catalogue import PlatformCatalogue
from edge_ai_deployment_pipeline.config.task import apply_task_to_config, resolve_task
from edge_ai_deployment_pipeline.core.types import (
    CvTask,
    DeploymentArtifact,
    FrameworkTag,
    LoadedModel,
    PlatformConfig,
    SelectedPlatform,
    UserRequirements,
)

logger = logging.getLogger(__name__)


def _resolve_yolo_calibration_yaml(raw: str, config_path: str | Path | None) -> str:
    """Resolve a project dataset yaml; leave Ultralytics built-in names (e.g. coco8.yaml) as-is."""
    path = Path(raw)
    if path.is_file():
        return str(path.resolve())
    resolved = _resolve_config_path(raw, config_path)
    if resolved.is_file():
        return str(resolved)
    return raw


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
        requirements: UserRequirements | None = None,
    ) -> DeploymentArtifact:
        board_id = selected.platform_id
        config = self._load_board_config(
            board_id,
            platform_config_path,
            toolchain=selected.toolchain,
            loaded=loaded,
            input_shape=input_shape,
            task_override=requirements.task if requirements else None,
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

        if adapter_id == "mcu" and loaded.framework != FrameworkTag.KERAS and loaded.framework != FrameworkTag.ULTRALYTICS:
            raise ValueError(
                f"MCU deployment ({board_id}) requires a Keras model (.keras or .h5). "
                f"Got {loaded.framework.value} model: {loaded.source_path}"
            )
        if loaded.framework == FrameworkTag.ULTRALYTICS:
            if resolve_task(config) != CvTask.OBJECT_DETECTION:
                raise ValueError(
                    f"Ultralytics deployment ({board_id}) requires "
                    f"--task object_detection (got {resolve_task(config).value})"
                )
            if adapter_id not in ("arm", "jetson", "mcu"):
                raise ValueError(
                    f"Ultralytics models are supported on ARM (ncnn), MCU and Jetson (tensorrt); "
                    f"got adapter '{adapter_id}'"
                )
        if adapter_id == "fpga" and loaded.framework != FrameworkTag.PYTORCH:
            raise ValueError(
                f"FPGA deployment ({board_id}) requires a PyTorch model (.pt). "
                f"Got {loaded.framework.value} model: {loaded.source_path}"
            )

        if not adapter.validate(loaded.model, config):
            hint = ""
            if adapter_id == "fpga" and loaded.framework != FrameworkTag.PYTORCH:
                hint = " Vitis-AI FPGA deployment supports PyTorch (.pt) models only."
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
            elif adapter_id == "mcu" and loaded.framework != FrameworkTag.KERAS:
                hint = (
                    f" MCU (STM32N6) requires a Keras model (.keras or .h5); "
                    f"got {loaded.framework.value}."
                )
            elif loaded.framework == FrameworkTag.ULTRALYTICS:
                if resolve_task(config) != CvTask.OBJECT_DETECTION:
                    hint = " Ultralytics deployment requires --task object_detection."
                elif adapter_id not in ("arm", "jetson"):
                    hint = " Ultralytics is supported on ARM (ncnn) and Jetson (tensorrt) only."
            raise ValueError(f"Adapter validation failed for board '{board_id}'.{hint}")

        converted = adapter.convert(loaded.model, config)
        artifact = adapter.build_artifact(converted, config)
        artifact.metadata["board_id"] = board_id
        artifact.metadata["platform_family"] = selected.platform_family
        artifact.metadata["task"] = str(config["task"])
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
        task_override: CvTask | None = None,
    ) -> PlatformConfig:
        catalogue = PlatformCatalogue.load(config_path)
        resolved_tc = toolchain or catalogue.list_toolchains(board_id)[0]
        spec = catalogue.resolve_board_config(board_id, resolved_tc)
        apply_task_to_config(spec, override=task_override)
        spec["output_dir"] = str(self.output_dir)
        spec["platform_id"] = board_id
        spec["board_id"] = board_id
        if input_shape is not None:
            spec["input_shape"] = input_shape
        if loaded is not None:
            spec["framework"] = loaded.framework.value
            spec["source_path"] = loaded.source_path
            spec["original_format"] = loaded.original_format
            if loaded.architecture_spec:
                spec["architecture_spec"] = loaded.architecture_spec
            if loaded.model_kwargs:
                spec["model_kwargs"] = loaded.model_kwargs
        if spec.get("calibration_data"):
            spec["calibration_data"] = str(
                _resolve_config_path(str(spec["calibration_data"]), config_path)
            )
        export_cfg = spec.get("yolo_export")
        if isinstance(export_cfg, dict) and export_cfg.get("calibration_data"):
            spec["yolo_export"] = {
                **export_cfg,
                "calibration_data": _resolve_yolo_calibration_yaml(
                    str(export_cfg["calibration_data"]),
                    config_path,
                ),
            }
        return spec
