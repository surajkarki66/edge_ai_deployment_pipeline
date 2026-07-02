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
}

# Pinned release tags for reproducible clones (override via config["stm32n6_repo_refs"]).
APPLICATION_REPO_REFS: dict[str, str] = {
    APPLICATION_IMAGE_CLASSIFICATION: "v2.3.0",
    APPLICATION_OBJECT_DETECTION: "v2.3.0",
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


def _resolve_repo_ref(config: dict[str, Any], application: str) -> str:
    refs = config.get("stm32n6_repo_refs")
    if isinstance(refs, dict):
        override = refs.get(application)
        if override:
            return str(override)
    return APPLICATION_REPO_REFS[application]


def _git_rev_parse(repo_dir: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _repo_matches_ref(dest: Path, ref: str) -> bool:
    if not dest.is_dir() or not (dest / ".git").is_dir():
        return False
    expected = _git_rev_parse(dest, "--verify", f"{ref}^{{commit}}")
    if expected is None:
        expected = _git_rev_parse(dest, "--verify", ref)
    if expected is None:
        return False
    head = _git_rev_parse(dest, "HEAD")
    return head == expected


def _clone_repo(url: str, dest: Path, *, ref: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        import shutil

        shutil.rmtree(dest)
    logger.info("Cloning STM32N6 reference project %s @ %s -> %s", url, ref, dest)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ref, url, str(dest)],
        check=True,
    )
    resolved = _git_rev_parse(dest, "HEAD")
    if resolved:
        logger.info("STM32N6 reference project at commit %s", resolved)


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
    ref = _resolve_repo_ref(config, application)
    if not _repo_matches_ref(dest, ref):
        _clone_repo(APPLICATION_REPOS[application], dest, ref=ref)
    return dest.resolve()