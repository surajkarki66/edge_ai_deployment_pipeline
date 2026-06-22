"""Resolve STM32N6 CubeIDE reference projects by application type."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

APPLICATION_IMAGE_CLASSIFICATION = "image_classification"
APPLICATION_OBJECT_DETECTION = "object_detection"
COMPILATION_OBJECT_DETECTION = "compilation_object_detection"

SUPPORTED_APPLICATIONS = frozenset(
    {APPLICATION_IMAGE_CLASSIFICATION, APPLICATION_OBJECT_DETECTION}
)

APPLICATION_REPOS: dict[str, str] = {
    APPLICATION_IMAGE_CLASSIFICATION: (
        "https://github.com/STMicroelectronics/"
        "STM32N6-GettingStarted-ImageClassification.git"
    ),
    APPLICATION_OBJECT_DETECTION: (
        "https://github.com/STMicroelectronics/"
        "STM32N6-GettingStarted-ObjectDetection.git"
    ),
    COMPILATION_OBJECT_DETECTION: (
        "https://github.com/surajkarki66/"
        "stm32ai-n6-yolov8-objectdetection.git"
    ),
}


def _repo_cache_root(config: dict[str, Any]) -> Path:
    if config.get("stm32n6_repos_dir"):
        return Path(str(config["stm32n6_repos_dir"])).resolve()
    env = os.environ.get("STM32N6_REPOS_DIR")
    if env:
        return Path(env).resolve()
    output = config.get("output_dir")
    if output:
        return Path(str(output)).resolve() / "_stm32n6_repos"
    return Path.cwd() / "stm32n6_repos"


def _repo_dir_name(application: str) -> str:
    url = APPLICATION_REPOS[application]
    name = url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name


def _clone_repo(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        import shutil

        shutil.rmtree(dest)
    logger.info("Cloning STM32N6 reference project %s -> %s", url, dest)
    subprocess.run(
        ["git", "clone", url, str(dest)],
        check=True,
    )


def resolve_application(config: dict[str, Any]) -> str | None:
    """Return a supported application key, or None if not specified."""
    raw = config.get("task") or config.get("application") or config.get("use_case")
    if not raw:
        return None
    key = str(raw).strip().lower().replace("-", "_")
    if key == "classification":
        key = APPLICATION_IMAGE_CLASSIFICATION
    elif key == "detection":
        key = APPLICATION_OBJECT_DETECTION
    if key not in SUPPORTED_APPLICATIONS:
        supported = ", ".join(sorted(SUPPORTED_APPLICATIONS))
        raise ValueError(f"Unknown task/application '{raw}'. Supported: {supported}")
    return key


def resolve_c_project(config: dict[str, Any]) -> Path:
    """Local clone path for the STM32N6 Getting Started project.

    Uses ``c_project_path`` when set; otherwise clones the GitHub repo for
    ``application`` (``image_classification`` or ``object_detection``).
    """
    explicit = config.get("c_project_path")
    if explicit:
        path = Path(str(explicit)).resolve()
        if not path.is_dir() or not any(path.iterdir()):
            raise FileNotFoundError(f"c_project_path not found or empty: {path}")
        return path

    application = resolve_application(config)
    if not application:
        supported = ", ".join(sorted(SUPPORTED_APPLICATIONS))
        raise ValueError(
            "STM32N6 compile requires 'application' "
            f"({supported}) or an explicit 'c_project_path'"
        )

    cache_root = _repo_cache_root(config)
    dest = cache_root / _repo_dir_name(application)
    if not dest.is_dir() or not any(dest.iterdir()):
        _clone_repo(APPLICATION_REPOS[application], dest)
        _clone_repo(APPLICATION_REPOS["compilation_object_detection"], cache_root / _repo_dir_name("compilation_object_detection"))
    return dest.resolve()