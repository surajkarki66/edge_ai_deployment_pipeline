from __future__ import annotations

import logging
import shutil

from datetime import datetime
from pathlib import Path

from edge_ai_deployment_pipeline.config.paths import DEFAULT_ARTIFACTS_DIR
from edge_ai_deployment_pipeline.core.logging_config import DEFAULT_LOG_DIR

logger = logging.getLogger(__name__)

DEFAULT_LOG_RETENTION = 50
DEFAULT_ARTIFACT_RUN_RETENTION = 20


def new_run_id() -> str:
    """Unique id for one pipeline invocation (timestamp + microseconds)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def run_artifacts_dir(artifacts_base: Path, run_id: str) -> Path:
    return artifacts_base / run_id


def _sorted_run_dirs(parent: Path) -> list[Path]:
    if not parent.is_dir():
        return []
    return sorted(
        (p for p in parent.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )


def _sorted_log_files(log_dir: Path) -> list[Path]:
    if not log_dir.is_dir():
        return []
    return sorted(
        (p for p in log_dir.glob("pipeline_*.log") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )


def _is_run_id_dir(name: str) -> bool:
    """Run ids look like 20260624_135204_423871."""
    parts = name.split("_")
    return len(parts) == 3 and parts[0].isdigit() and len(parts[0]) == 8


def rotate_logs(log_dir: Path | None = None, *, keep: int = DEFAULT_LOG_RETENTION) -> int:
    """Delete oldest pipeline log files beyond *keep*. Returns number removed."""
    directory = log_dir or DEFAULT_LOG_DIR
    files = _sorted_log_files(directory)
    removed = 0
    for path in files[: max(0, len(files) - keep)]:
        path.unlink(missing_ok=True)
        removed += 1
    if removed:
        logger.debug("Rotated %d log file(s) from %s", removed, directory)
    return removed


def rotate_artifact_runs(
    artifacts_base: Path,
    *,
    keep: int = DEFAULT_ARTIFACT_RUN_RETENTION,
) -> int:
    """Delete oldest run directories under *artifacts_base*. Returns number removed."""
    run_dirs = [p for p in _sorted_run_dirs(artifacts_base) if _is_run_id_dir(p.name)]
    removed = 0
    for path in run_dirs[: max(0, len(run_dirs) - keep)]:
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    if removed:
        logger.debug("Rotated %d artifact run(s) from %s", removed, artifacts_base)
    return removed


def clean_workspace(
    *,
    artifacts_base: Path | None = None,
    log_dir: Path | None = None,
    keep_logs: int = 0,
    keep_runs: int = 0,
) -> None:
    """Remove artifact runs and/or logs."""
    base = artifacts_base or DEFAULT_ARTIFACTS_DIR
    logs = log_dir or DEFAULT_LOG_DIR

    if base.is_dir():
        run_dirs = [p for p in _sorted_run_dirs(base) if _is_run_id_dir(p.name)]
        for path in run_dirs[: max(0, len(run_dirs) - keep_runs)]:
            shutil.rmtree(path, ignore_errors=True)
        for path in list(base.iterdir()):
            if path.is_dir() and not _is_run_id_dir(path.name):
                shutil.rmtree(path, ignore_errors=True)

    if logs.is_dir():
        if keep_logs <= 0:
            for path in _sorted_log_files(logs):
                path.unlink(missing_ok=True)
        else:
            rotate_logs(logs, keep=keep_logs)
