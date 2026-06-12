#!/usr/bin/env python3
from __future__ import annotations

import cv2
import logging
import torch
import argparse
import glob
import math
import os
import random


from pathlib import Path
from tqdm import tqdm
from pytorch_nndct.apis import torch_quantizer

logger = logging.getLogger(__name__)

random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="model.pt", help="PyTorch weights file (state_dict).")
parser.add_argument(
    "--model_class",
    default=None,
    help="Network definition inside the workspace, e.g. models/model_arch.py:Net",
)
parser.add_argument(
    "--model_kwargs",
    default="{}",
    help="JSON object passed to the model constructor",
)
parser.add_argument("--config_file", default=None, help="Quantization config file.")
parser.add_argument("--subset_len", default=None, type=int, help="Calibration subset length.")
parser.add_argument("--batch_size", default=32, type=int, help="Calibration batch size.")
parser.add_argument(
    "--quant_mode",
    default="calib",
    choices=["float", "calib", "test"],
    help="float | calib | test (use test with --deploy for xmodel).",
)
parser.add_argument("--deploy", action="store_true", help="Export xmodel for deployment.")
parser.add_argument("--inspect", action="store_true", help="Inspect model.")
parser.add_argument("--target", default="", help="DPU target, e.g. DPUCZDX8G_ISA1_B4096.")
parser.add_argument("--channels", default=3, type=int, help="Input channels (1 or 3).")
parser.add_argument("--img_height", default=224, type=int, help="Input height.")
parser.add_argument("--img_width", default=224, type=int, help="Input width.")
parser.add_argument(
    "--calibration_list",
    default="data/val_ids.txt",
    help="Image list file or directory for calibration.",
)
args, _ = parser.parse_known_args()

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}


class LoadImages:
    """Load images in batches from a path, directory, or list file."""

    def __init__(self, path: str, batch: int = 1):
        parent = None
        if isinstance(path, str) and Path(path).suffix == ".txt":
            parent = Path(path).parent
            path = Path(path).read_text().splitlines()
        files: list[str] = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, "*.*"))))
            elif os.path.isfile(a):
                files.append(a)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))
            else:
                raise FileNotFoundError(f"{p} does not exist")
        images = [f for f in files if f.split(".")[-1].lower() in IMG_FORMATS]
        self.files = images
        self.nf = len(images)
        self.bs = batch
        if self.nf == 0:
            raise FileNotFoundError(f"No images found under {path}")

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        paths, imgs = [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:
                if imgs:
                    return paths, imgs
                raise StopIteration
            path = self.files[self.count]
            im0 = cv2.imread(path)
            if im0 is None:
                raise FileNotFoundError(f"Image not found: {path}")
            paths.append(path)
            imgs.append(im0)
            self.count += 1
        return paths, imgs

    def __len__(self):
        return math.ceil(self.nf / self.bs)


def _import_model_class(spec: str) -> type:
    import importlib.util
    import sys

    if ":" not in spec:
        raise ValueError(f"Invalid --model_class {spec!r}")
    module_part, class_name = spec.rsplit(":", 1)
    path = Path(module_part)
    if not path.is_file():
        raise FileNotFoundError(f"Model definition not found: {path}")

    module_name = "vitis_model_arch"
    spec_obj = importlib.util.spec_from_file_location(module_name, path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"Cannot import model class from {path}")
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    spec_obj.loader.exec_module(module)
    model_cls = getattr(module, class_name)
    return model_cls


def _extract_state_dict(obj) -> dict:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        raise TypeError(f"Unsupported checkpoint dict at {args.model_path}")
    raise TypeError(f"Expected state_dict checkpoint, got {type(obj).__name__}")


def load_model(model_path: str) -> torch.nn.Module:
    import json

    if not args.model_class:
        raise ValueError("--model_class is required (e.g. data/model_class/cifar_net.py:Net)")

    model_cls = _import_model_class(args.model_class)
    kwargs = json.loads(args.model_kwargs or "{}")
    if not isinstance(kwargs, dict):
        raise ValueError("--model_kwargs must be a JSON object")

    model = model_cls(**kwargs)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module.") :]: value
            for key, value in state_dict.items()
            if key.startswith("module.")
        }
    model.load_state_dict(state_dict)
    model.float()
    model.eval()
    return model


def _preprocess_image(img) -> torch.Tensor:
    h, w = args.img_height, args.img_width
    if args.channels == 1:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(img, (w, h))
        tensor = torch.from_numpy(resized).unsqueeze(0).float() / 255.0
    else:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(img, (w, h))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    return tensor.to(device)


def run_inference(model: torch.nn.Module) -> None:
    dataset = LoadImages(args.calibration_list, args.batch_size)
    for batch in tqdm(dataset, total=len(dataset)):
        _, imgs = batch
        batch_tensor = torch.stack([_preprocess_image(img) for img in imgs])
        _ = model(batch_tensor)
    logger.info("Calibration inference done.")


def quantization() -> None:
    quant_mode = args.quant_mode
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len
    config_file = args.config_file
    target = args.target or None

    model = load_model(args.model_path)

    if quant_mode != "test" and deploy:
        deploy = False
        logger.warning("xmodel export requires quant_mode=test; deploy disabled.")
    if deploy and (batch_size != 1 or subset_len not in (None, 1)):
        logger.warning("xmodel export needs batch_size=1; adjusting.")
        batch_size = 1
        subset_len = 1

    example = torch.randn([batch_size, args.channels, args.img_height, args.img_width])
    quantizer = torch_quantizer(
        quant_mode,
        model,
        (example,),
        device=device,
        quant_config_file=config_file,
        target=target,
    )
    quant_model = quantizer.quant_model
    run_inference(quant_model)

    if quant_mode == "calib":
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel()
        logger.info("Exported xmodel to quantize_result/")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("-------- Start quantization: %s (%s)", args.model_path, args.quant_mode)
    quantization()
    logger.info("-------- End quantization: %s", args.model_path)
