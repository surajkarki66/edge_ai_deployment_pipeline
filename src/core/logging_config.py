from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = _REPO_ROOT / "logs"


def _unique_log_filename() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"pipeline_{stamp}.log"


def setup_logging(
    *,
    verbose: bool = False,
    log_dir: Path | str | None = None,
    log_file: str | None = None,
) -> Path:
    """Send pipeline logs to stderr and a new ``logs/pipeline_<timestamp>.log`` file."""
    directory = Path(log_dir or DEFAULT_LOG_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / (log_file or _unique_log_filename())

    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root.addHandler(console)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info("Logging to %s", log_path)
    return log_path
