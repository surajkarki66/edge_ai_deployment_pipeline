"""CV task resolution for platform deployment config."""

from __future__ import annotations

from typing import Any

from edge_ai_deployment_pipeline.core.types import CvTask, PlatformConfig

_TASK_ALIASES: dict[str, CvTask] = {
    "image_classification": CvTask.IMAGE_CLASSIFICATION,
    "classification": CvTask.IMAGE_CLASSIFICATION,
    "image-classification": CvTask.IMAGE_CLASSIFICATION,
    "object_detection": CvTask.OBJECT_DETECTION,
    "detection": CvTask.OBJECT_DETECTION,
    "object-detection": CvTask.OBJECT_DETECTION,
}


def normalize_task_key(raw: str) -> str:
    """Normalize user/config task strings to canonical enum values."""
    key = str(raw).strip().lower().replace("-", "_")
    if key == "classification":
        return CvTask.IMAGE_CLASSIFICATION.value
    if key == "detection":
        return CvTask.OBJECT_DETECTION.value
    return key


def parse_task(raw: str | CvTask) -> CvTask:
    if isinstance(raw, CvTask):
        return raw
    key = normalize_task_key(str(raw))
    if key in {t.value for t in CvTask}:
        return CvTask(key)
    alias = _TASK_ALIASES.get(key) or _TASK_ALIASES.get(str(raw).strip().lower())
    if alias is not None:
        return alias
    supported = ", ".join(t.value for t in CvTask)
    raise ValueError(f"Unknown task '{raw}'. Supported: {supported}")


def resolve_task(
    config: PlatformConfig,
    *,
    override: str | CvTask | None = None,
) -> CvTask:
    """Resolve effective CV task: CLI/user override > merged platform config."""
    if override is not None:
        return parse_task(override)

    for key in ("task", "application", "use_case"):
        raw = config.get(key)
        if raw:
            return parse_task(str(raw))

    return CvTask.IMAGE_CLASSIFICATION


def apply_task_to_config(
    config: PlatformConfig,
    *,
    override: str | CvTask | None = None,
) -> CvTask:
    """Write canonical ``task`` (and STM32 ``application`` alias) into *config*."""
    task = resolve_task(config, override=override)
    config["task"] = task.value
    # STM32N6 tooling still keys off ``application``.
    config["application"] = task.value
    return task
