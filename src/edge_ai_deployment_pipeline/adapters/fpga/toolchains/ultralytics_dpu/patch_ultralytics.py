"""Copy DPU-compatible layers into the installed ultralytics package.

Used before Vitis-AI PTQ for Ultralytics YOLO object detection. Replaces
``ultralytics/nn/modules/head.py`` and ``block.py`` while leaving ``conv.py``
and ``transformer.py`` from the pinned ultralytics install.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

from pathlib import Path

logger = logging.getLogger(__name__)

VITIS_ULTRALYTICS_VERSION = "8.4.24"
CUSTOM_LAYERS_DIR = Path(__file__).resolve().parent / "custom_layers"


def patch_ultralytics(custom_layers_dir: Path | str | None = None) -> Path:
    """Overwrite ultralytics ``nn/modules`` files from *custom_layers_dir*."""
    import ultralytics

    src_dir = Path(custom_layers_dir or CUSTOM_LAYERS_DIR).resolve()
    ultra_path = Path(ultralytics.__file__).resolve().parent
    dst_dir = ultra_path / "nn" / "modules"

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Custom layers folder not found: {src_dir}")
    if not dst_dir.is_dir():
        raise FileNotFoundError(f"Ultralytics nn/modules not found: {dst_dir}")

    logger.info("Ultralytics package: %s", ultra_path)
    logger.info("Patching from: %s", src_dir)

    patched: list[str] = []
    for src_file in sorted(src_dir.iterdir()):
        if not src_file.is_file() or src_file.suffix != ".py":
            continue
        dst_file = dst_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        patched.append(src_file.name)
        logger.info("Copied %s -> %s", src_file.name, dst_dir)

    if not patched:
        raise FileNotFoundError(f"No .py files found in {src_dir}")

    return dst_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Patch ultralytics for Vitis-AI DPU export")
    parser.add_argument(
        "custom_layers_dir",
        nargs="?",
        default=os.environ.get("VITIS_CUSTOM_LAYERS_DIR", "/workspace/custom_layers"),
        help="Directory with head.py and block.py (default: /workspace/custom_layers)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        patch_ultralytics(args.custom_layers_dir)
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    logger.info("All custom layer files copied successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
