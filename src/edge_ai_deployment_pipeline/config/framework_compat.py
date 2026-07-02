from __future__ import annotations

from edge_ai_deployment_pipeline.core.types import CvTask, FrameworkTag

_PLATFORM_FRAMEWORKS: dict[str, frozenset[FrameworkTag]] = {
    "arm": frozenset({FrameworkTag.PYTORCH, FrameworkTag.KERAS, FrameworkTag.ULTRALYTICS}),
    "jetson": frozenset({FrameworkTag.PYTORCH, FrameworkTag.KERAS, FrameworkTag.ULTRALYTICS}),
    "fpga": frozenset({FrameworkTag.PYTORCH, FrameworkTag.ULTRALYTICS}),
    "mcu": frozenset({FrameworkTag.KERAS, FrameworkTag.ULTRALYTICS}),
}

_PLATFORM_LABELS: dict[str, str] = {
    "arm": "ARM (ncnn)",
    "jetson": "Jetson (TensorRT)",
    "fpga": "FPGA (Vitis AI)",
    "mcu": "MCU (STM32N6)",
}

_FRAMEWORK_LABELS: dict[FrameworkTag, str] = {
    FrameworkTag.PYTORCH: "PyTorch weights (.pt/.pth + --model-class)",
    FrameworkTag.KERAS: "Keras (.keras/.h5)",
    FrameworkTag.ULTRALYTICS: "Ultralytics checkpoint (.pt)",
}


def framework_platform_incompatibility_reason(
    framework: FrameworkTag,
    task: CvTask,
    adapter_id: str,
) -> str | None:
    """Return a human-readable rejection reason, or None if the combo is supported."""
    family = adapter_id
    allowed = _PLATFORM_FRAMEWORKS.get(family)
    if allowed is None:
        return f"unknown platform adapter '{adapter_id}'"

    if framework not in allowed:
        platform_label = _PLATFORM_LABELS.get(family, family)
        framework_label = _FRAMEWORK_LABELS.get(framework, framework.value)
        if family == "fpga" and framework == FrameworkTag.KERAS:
            return (
                f"{platform_label} requires a PyTorch or Ultralytics model; "
                f"got {framework_label}"
            )
        if family == "mcu" and framework == FrameworkTag.PYTORCH:
            return (
                f"{platform_label} supports Keras and Ultralytics models only; "
                f"got {framework_label}"
            )
        accepted = ", ".join(
            _FRAMEWORK_LABELS.get(tag, tag.value) for tag in sorted(allowed, key=lambda t: t.value)
        )
        return (
            f"{platform_label} does not accept {framework_label}; "
            f"supported: {accepted}"
        )

    if framework == FrameworkTag.ULTRALYTICS and task != CvTask.OBJECT_DETECTION:
        return (
            "Ultralytics models require --task object_detection "
            f"(got {task.value})"
        )

    if (
        family == "mcu"
        and framework == FrameworkTag.KERAS
        and task == CvTask.OBJECT_DETECTION
    ):
        return (
            "MCU Keras deployment supports image_classification only; "
            "use --task object_detection with an Ultralytics checkpoint for detection"
        )

    return None
