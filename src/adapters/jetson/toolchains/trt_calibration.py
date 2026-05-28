"""Load image folders for TensorRT INT8 calibration."""

from __future__ import annotations

import logging

from pathlib import Path

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_calibration_images(calibration_data: Path) -> list[Path]:
    """Return image paths from a directory or a ``.txt`` list file."""
    if calibration_data.is_file() and calibration_data.suffix == ".txt":
        lines = [
            ln.strip()
            for ln in calibration_data.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        paths: list[Path] = []
        list_parent = calibration_data.parent
        for line in lines:
            p = Path(line)
            if not p.is_absolute():
                p = (list_parent / p).resolve()
            if p.is_file():
                paths.append(p)
            else:
                logger.warning("Skipping missing calibration image: %s", p)
        return paths
    if calibration_data.is_dir():
        return sorted(
            p
            for p in calibration_data.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
    raise FileNotFoundError(
        f"calibration_data must be a .txt list or image directory: {calibration_data}"
    )


def _load_rgb_array(path: Path, *, height: int, width: int):
    try:
        from PIL import Image
    except ImportError:
        import numpy as np
        import tensorflow as tf

        img = tf.keras.utils.load_img(str(path), target_size=(height, width))
        return np.asarray(img, dtype=np.float32) / 255.0

    import numpy as np

    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB").resize((width, height)), dtype=np.float32)
    return arr / 255.0


class TensorRTCalibrationLoader:
    """Yield preprocessed float32 batches on CUDA for ``IInt8Calibrator``."""

    def __init__(
        self,
        calibration_data: Path | str,
        shape: tuple[int, int, int, int],
        *,
        channels_last: bool = True,
        batch_size: int = 1,
        device: str = "cuda",
    ) -> None:
        import torch

        self.batch_size = max(1, int(batch_size))
        self.device = torch.device(device)
        self.channels_last = channels_last
        self.paths = collect_calibration_images(Path(calibration_data).resolve())
        if not self.paths:
            raise FileNotFoundError(f"No calibration images found under {calibration_data}")

        n, d1, d2, d3 = shape
        if channels_last and d3 <= 4 and d1 > d3:
            self._layout = "nhwc"
            self._height, self._width, self._channels = d1, d2, d3
        else:
            self._layout = "nchw"
            self._channels, self._height, self._width = d1, d2, d3
        self._batch = n if n > 0 else 1
        logger.info(
            "TensorRT calibration: %d images, layout=%s, shape=%s",
            len(self.paths),
            self._layout,
            shape,
        )

    def __iter__(self):
        import torch

        for start in range(0, len(self.paths), self.batch_size):
            paths = self.paths[start : start + self.batch_size]
            arrays = [
                _load_rgb_array(p, height=self._height, width=self._width) for p in paths
            ]
            import numpy as np

            batch = np.stack(arrays, axis=0)
            if self._layout == "nhwc":
                tensor = torch.from_numpy(batch).to(self.device)
            else:
                tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(self.device)
            if tensor.shape[0] != self._batch:
                if self._layout == "nhwc":
                    pad = torch.zeros(
                        self._batch - tensor.shape[0],
                        self._height,
                        self._width,
                        self._channels,
                        device=self.device,
                    )
                else:
                    pad = torch.zeros(
                        self._batch - tensor.shape[0],
                        self._channels,
                        self._height,
                        self._width,
                        device=self.device,
                    )
                tensor = torch.cat([tensor, pad], dim=0)
            yield tensor.float()
