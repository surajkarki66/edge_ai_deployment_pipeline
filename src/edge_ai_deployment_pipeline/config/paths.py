"""Shared filesystem paths for the deployment pipeline."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
DEFAULT_ARTIFACTS_DIR = DATA_DIR / "artifacts"
DEFAULT_MODEL_CLASS_DIR = DATA_DIR / "model_class"
DEFAULT_CALIBRATION_DIR = DATA_DIR / "calibration"

EXAMPLE_MODEL_CLASS = f"{DEFAULT_MODEL_CLASS_DIR.as_posix()}/mnist_net.py:Net"
