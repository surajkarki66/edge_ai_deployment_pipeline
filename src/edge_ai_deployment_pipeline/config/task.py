from __future__ import annotations

from edge_ai_deployment_pipeline.core.types import CvTask, PlatformConfig


def normalize_task_key(raw: str) -> str:
    """Normalize task strings to canonical enum values."""
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
    if key not in {t.value for t in CvTask}:
        supported = ", ".join(t.value for t in CvTask)
        raise ValueError(f"Unknown task '{raw}'. Supported: {supported}")
    return CvTask(key)


_TASK_REQUIRED_MSG = (
    "CV task is required. Pass --task image_classification or --task object_detection."
)


def resolve_task(
    config: PlatformConfig,
    *,
    override: str | CvTask | None = None,
) -> CvTask:
    """Resolve effective CV task from CLI override or runtime config (set by apply_task_to_config)."""
    if override is not None:
        return parse_task(override)

    for key in ("task", "application", "use_case"):
        raw = config.get(key)
        if raw:
            return parse_task(str(raw))

    raise ValueError(_TASK_REQUIRED_MSG)


def apply_task_to_config(
    config: PlatformConfig,
    *,
    override: str | CvTask | None = None,
) -> CvTask:
    """Write canonical ``task`` (and STM32 ``application`` alias) into *config*."""
    if override is None:
        raise ValueError(_TASK_REQUIRED_MSG)
    task = parse_task(override)
    config["task"] = task.value
    # STM32N6 tooling still keys off ``application``.
    config["application"] = task.value
    return task
