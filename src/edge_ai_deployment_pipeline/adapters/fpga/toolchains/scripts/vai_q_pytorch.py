#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cv2
import glob
import json
import logging
import math
import os
import random
import sys

from pathlib import Path

import torch
from tqdm import tqdm
from pytorch_nndct.apis import torch_quantizer


# =============================================================================
# Logging / Random Seeds
# =============================================================================

logger = logging.getLogger(__name__)

random.seed(0)
torch.manual_seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Generic Vitis AI PyTorch quantization script for "
                "classification and object detection."
)

parser.add_argument(
    "--model_path",
    default="model.pt",
    help="PyTorch weights/checkpoint file."
)

parser.add_argument(
    "--model_class",
    default=None,
    help=(
        "Classification model definition. "
        "Example: models/model_arch.py:Net. "
        "If omitted, the script assumes an object-detection checkpoint."
    ),
)

parser.add_argument(
    "--model_kwargs",
    default="{}",
    help="JSON object passed to the classification model constructor.",
)

parser.add_argument(
    "--config_file",
    default=None,
    help="Vitis AI quantization config file.",
)

parser.add_argument(
    "--subset_len",
    default=None,
    type=int,
    help="Number of calibration images to use. Default: all images.",
)

parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="Calibration batch size.",
)

parser.add_argument(
    "--quant_mode",
    default="calib",
    choices=["float", "calib", "test"],
    help=(
        "Quantization mode: float, calib, or test. "
        "Use test together with --deploy for xmodel export."
    ),
)

parser.add_argument(
    "--deploy",
    action="store_true",
    help="Export TorchScript, ONNX and XModel.",
)

parser.add_argument(
    "--inspect",
    action="store_true",
    help="Inspect model.",
)

parser.add_argument(
    "--target",
    default="",
    help="DPU target, e.g. DPUCZDX8G_ISA1_B4096.",
)

parser.add_argument(
    "--channels",
    default=3,
    type=int,
    help="Input channels. Classification MNIST uses 1, YOLO normally uses 3.",
)

parser.add_argument(
    "--img_height",
    default=224,
    type=int,
    help="Input image height.",
)

parser.add_argument(
    "--img_width",
    default=224,
    type=int,
    help="Input image width.",
)

parser.add_argument(
    "--calibration_list",
    default="data/val_ids.txt",
    help="Calibration image list, directory, image or glob.",
)

args, _ = parser.parse_known_args()


# =============================================================================
# Supported image formats
# =============================================================================

IMG_FORMATS = {
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
}


# =============================================================================
# Image Loader
# =============================================================================

class LoadImages:
    """
    Load images from:

        - directory
        - image file
        - glob pattern
        - .txt file containing image paths

    Images are returned in batches.
    """

    def __init__(self, path: str, batch: int = 1, subset_len=None):

        parent = None

        # -------------------------------------------------------------
        # .txt file
        # -------------------------------------------------------------

        if isinstance(path, str) and Path(path).suffix.lower() == ".txt":

            parent = Path(path).parent

            path = Path(path).read_text().splitlines()

        files = []

        # -------------------------------------------------------------
        # Resolve paths
        # -------------------------------------------------------------

        path_list = (
            sorted(path)
            if isinstance(path, (list, tuple))
            else [path]
        )

        for p in path_list:

            a = str(Path(p).absolute())

            # Glob
            if "*" in a:

                files.extend(
                    sorted(
                        glob.glob(
                            a,
                            recursive=True
                        )
                    )
                )

            # Directory
            elif os.path.isdir(a):

                files.extend(
                    sorted(
                        glob.glob(
                            os.path.join(a, "*.*")
                        )
                    )
                )

            # File
            elif os.path.isfile(a):

                files.append(a)

            # Relative path from txt file
            elif parent is not None and (parent / p).is_file():

                files.append(
                    str(
                        (parent / p).absolute()
                    )
                )

            else:

                raise FileNotFoundError(
                    f"{p} does not exist"
                )

        # -------------------------------------------------------------
        # Filter image files
        # -------------------------------------------------------------

        images = [
            f
            for f in files
            if Path(f).suffix.lower().replace(".", "") in IMG_FORMATS
        ]

        # Remove duplicates while keeping order
        images = list(dict.fromkeys(images))

        # -------------------------------------------------------------
        # Apply subset length
        # -------------------------------------------------------------

        if subset_len is not None:

            if subset_len <= 0:
                raise ValueError(
                    "--subset_len must be greater than 0"
                )

            images = images[:subset_len]

        self.files = images
        self.nf = len(images)
        self.bs = batch

        if self.nf == 0:

            raise FileNotFoundError(
                f"No images found under {path}"
            )

    def __iter__(self):

        self.count = 0

        return self

    def __next__(self):

        paths = []
        imgs = []

        while len(imgs) < self.bs:

            if self.count >= self.nf:

                if imgs:

                    return paths, imgs

                raise StopIteration

            path = self.files[self.count]

            im0 = cv2.imread(path)

            if im0 is None:

                raise FileNotFoundError(
                    f"Could not read image: {path}"
                )

            paths.append(path)
            imgs.append(im0)

            self.count += 1

        return paths, imgs

    def __len__(self):

        return math.ceil(
            self.nf / self.bs
        )


# =============================================================================
# Model Class Import
# =============================================================================

def _import_model_class(spec: str) -> type:

    """
    Import model class from:

        path/to/model.py:ClassName

    Example:

        models/model_arch.py:Net
    """

    if ":" not in spec:

        raise ValueError(
            f"Invalid --model_class {spec!r}. "
            f"Expected format: path/to/model.py:ClassName"
        )

    module_part, class_name = spec.rsplit(":", 1)

    path = Path(module_part)

    if not path.is_file():

        raise FileNotFoundError(
            f"Model definition not found: {path}"
        )

    module_name = "vitis_model_arch"

    spec_obj = (
        __import__("importlib.util")
        .util.spec_from_file_location(
            module_name,
            path
        )
    )

    if spec_obj is None or spec_obj.loader is None:

        raise ImportError(
            f"Cannot import model class from {path}"
        )

    module = (
        __import__("importlib.util")
        .util.module_from_spec(spec_obj)
    )

    sys.modules[module_name] = module

    spec_obj.loader.exec_module(module)

    if not hasattr(module, class_name):

        raise AttributeError(
            f"Class '{class_name}' not found in {path}"
        )

    return getattr(module, class_name)


# =============================================================================
# Extract State Dictionary
# =============================================================================

def _extract_state_dict(obj) -> dict:

    if isinstance(obj, dict):

        # Common checkpoint formats
        for key in (
            "state_dict",
            "model_state_dict",
        ):

            if (
                key in obj
                and isinstance(obj[key], dict)
            ):

                return obj[key]

        # Direct state_dict
        if obj and all(
            isinstance(v, torch.Tensor)
            for v in obj.values()
        ):

            return obj

        raise TypeError(
            f"Unsupported checkpoint dictionary "
            f"format: {args.model_path}"
        )

    raise TypeError(
        f"Expected state_dict checkpoint, "
        f"got {type(obj).__name__}"
    )


# =============================================================================
# Load Model
# =============================================================================

def load_model(model_path: str) -> torch.nn.Module:

    # =================================================================
    # CLASSIFICATION
    #
    # --model_class is specified
    # =================================================================

    if args.model_class is not None:

        logger.info(
            "Model mode: CLASSIFICATION"
        )

        logger.info(
            "Model class: %s",
            args.model_class
        )

        # -------------------------------------------------------------
        # Import architecture
        # -------------------------------------------------------------

        model_cls = _import_model_class(
            args.model_class
        )

        # -------------------------------------------------------------
        # Constructor arguments
        # -------------------------------------------------------------

        kwargs = json.loads(
            args.model_kwargs or "{}"
        )

        if not isinstance(kwargs, dict):

            raise ValueError(
                "--model_kwargs must be a JSON object"
            )

        # -------------------------------------------------------------
        # Create model
        # -------------------------------------------------------------

        model = model_cls(**kwargs)

        # -------------------------------------------------------------
        # Load state_dict
        # -------------------------------------------------------------

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = _extract_state_dict(
            checkpoint
        )

        # Remove DataParallel "module." prefix
        if any(
            key.startswith("module.")
            for key in state_dict
        ):

            state_dict = {
                key[len("module."):]: value
                for key, value in state_dict.items()
            }

        model.load_state_dict(
            state_dict
        )

        model.float()
        model.eval()

        return model

    # =================================================================
    # OBJECT DETECTION
    # =================================================================

    else:

        logger.info(
            "Model mode: OBJECT DETECTION"
        )

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

        # -------------------------------------------------------------
        # Existing detection checkpoint format
        # -------------------------------------------------------------

        if isinstance(checkpoint, dict) and "model" in checkpoint:

            model = checkpoint["model"]

        else:

            # Also allow checkpoint itself to be a model
            model = checkpoint

        model.float()
        model.eval()

        return model


# =============================================================================
# Preprocessing
# =============================================================================

def _preprocess_classification(
    img,
) -> torch.Tensor:

    """
    MNIST preprocessing matching the training script:

        PIL MNIST image
             ↓
        ToTensor()
             ↓
        /255
             ↓
        Normalize(0.1307, 0.3081)

    Output:

        [1, 28, 28]
    """

    # -------------------------------------------------------------
    # Convert BGR -> grayscale
    # -------------------------------------------------------------

    if img.ndim == 3:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    # -------------------------------------------------------------
    # Resize to 28x28
    # -------------------------------------------------------------

    img = cv2.resize(
        img,
        (
            args.img_width,
            args.img_height
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # -------------------------------------------------------------
    # Convert uint8 [0,255]
    # -> float [0,1]
    # -------------------------------------------------------------

    tensor = (
        torch.from_numpy(img)
        .unsqueeze(0)
        .float()
        / 255.0
    )
    tensor = (
        tensor - 0.1307
    ) / 0.3081

    return tensor.to(device)


def _preprocess_detection(
    img,
) -> torch.Tensor:

    """
    Existing generic detection preprocessing.

    BGR image
        ↓
    resize
        ↓
    [0,255] -> [0,1]
        ↓
    CHW
    """

    h = args.img_height
    w = args.img_width

    if args.channels == 1:

        if img.ndim == 3:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

        resized = cv2.resize(
            img,
            (w, h)
        )

        tensor = (
            torch.from_numpy(resized)
            .unsqueeze(0)
            .float()
            / 255.0
        )

    else:

        if img.ndim == 2:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2BGR
            )

        resized = cv2.resize(
            img,
            (w, h)
        )

        tensor = (
            torch.from_numpy(
                resized.transpose(2, 0, 1)
            )
            .float()
            / 255.0
        )

    return tensor.to(device)


def _preprocess_image(
    img,
) -> torch.Tensor:

    """
    Automatically select preprocessing based on
    whether --model_class was specified.

    model_class specified:
        classification

    model_class not specified:
        object detection
    """

    if args.model_class is not None:

        return _preprocess_classification(
            img
        )

    else:

        return _preprocess_detection(
            img
        )


# =============================================================================
# Calibration Inference
# =============================================================================

def run_inference(
    model: torch.nn.Module,
) -> None:

    dataset = LoadImages(
        args.calibration_list,
        args.batch_size,
        args.subset_len,
    )

    logger.info(
        "Calibration images: %d",
        dataset.nf
    )

    logger.info(
        "Calibration batch size: %d",
        args.batch_size
    )

    # -------------------------------------------------------------
    # Print mode
    # -------------------------------------------------------------

    if args.model_class is not None:

        logger.info(
            "Preprocessing mode: MNIST classification"
        )

        logger.info(
            "Input: %d x %d x %d",
            args.channels,
            args.img_height,
            args.img_width,
        )

        logger.info(
            "Normalization: mean=0.1307, std=0.3081"
        )

    else:

        logger.info(
            "Preprocessing mode: object detection"
        )

    # -------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------

    for batch_idx, batch in enumerate(
        tqdm(
            dataset,
            total=len(dataset),
            desc="Calibration"
        )
    ):

        _, imgs = batch

        batch_tensor = torch.stack(
            [
                _preprocess_image(img)
                for img in imgs
            ]
        )

        # Forward pass
        _ = model(
            batch_tensor
        )

    logger.info(
        "Calibration inference done."
    )


# =============================================================================
# Quantization
# =============================================================================

def quantization() -> None:

    quant_mode = args.quant_mode
    deploy = args.deploy

    batch_size = args.batch_size
    subset_len = args.subset_len

    config_file = args.config_file

    target = args.target or None

    # -----------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------

    model = load_model(
        args.model_path
    )

    # -----------------------------------------------------------------
    # Print configuration
    # -----------------------------------------------------------------

    logger.info(
        "============================================================"
    )

    logger.info(
        "Quantization configuration"
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "Model: %s",
        args.model_path
    )

    logger.info(
        "Model class: %s",
        args.model_class
    )

    logger.info(
        "Channels: %d",
        args.channels
    )

    logger.info(
        "Input size: %d x %d",
        args.img_width,
        args.img_height
    )

    logger.info(
        "Batch size: %d",
        batch_size
    )

    logger.info(
        "Subset length: %s",
        subset_len
    )

    logger.info(
        "Quantization mode: %s",
        quant_mode
    )

    logger.info(
        "Deploy: %s",
        deploy
    )

    logger.info(
        "============================================================"
    )

    # -----------------------------------------------------------------
    # Deployment requires test mode
    # -----------------------------------------------------------------

    if quant_mode != "test" and deploy:

        deploy = False

        logger.warning(
            "XModel export requires --quant_mode test. "
            "--deploy has been disabled."
        )

    # -----------------------------------------------------------------
    # Deployment requires batch size 1
    # -----------------------------------------------------------------

    if deploy:

        if batch_size != 1:

            logger.warning(
                "XModel deployment requires batch_size=1. "
                "Changing batch size to 1."
            )

            batch_size = 1

        if subset_len not in (
            None,
            1,
        ):

            logger.warning(
                "XModel deployment requires subset_len=1. "
                "Changing subset_len to 1."
            )

            subset_len = 1

    # -----------------------------------------------------------------
    # Example input
    # -----------------------------------------------------------------

    example = torch.randn(
        [
            batch_size,
            args.channels,
            args.img_height,
            args.img_width,
        ],
        device=device,
    )

    logger.info(
        "Creating Vitis AI quantizer..."
    )

    # -----------------------------------------------------------------
    # Vitis AI quantizer
    # -----------------------------------------------------------------

    quantizer = torch_quantizer(
        quant_mode,
        model,
        (example,),
        device=device,
        quant_config_file=config_file,
        target=target,
    )

    quant_model = quantizer.quant_model

    # -----------------------------------------------------------------
    # Calibration / inference
    # -----------------------------------------------------------------

    run_inference(
        quant_model
    )

    # -----------------------------------------------------------------
    # Export quantization configuration
    # -----------------------------------------------------------------

    if quant_mode == "calib":

        logger.info(
            "Exporting quantization configuration..."
        )

        quantizer.export_quant_config()

        logger.info(
            "Quantization configuration exported."
        )

    # -----------------------------------------------------------------
    # Export deployment files
    # -----------------------------------------------------------------

    if deploy:

        logger.info(
            "Exporting TorchScript..."
        )

        quantizer.export_torch_script()

        logger.info(
            "Exporting ONNX..."
        )

        quantizer.export_onnx_model()

        logger.info(
            "Exporting XModel..."
        )

        quantizer.export_xmodel()

        logger.info(
            "============================================================"
        )

        logger.info(
            "XModel exported to quantize_result/"
        )

        logger.info(
            "============================================================"
        )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s: "
            "%(message)s"
        ),
    )

    logger.info(
        "-------- Start quantization: %s (%s) --------",
        args.model_path,
        args.quant_mode,
    )

    quantization()

    logger.info(
        "-------- End quantization: %s --------",
        args.model_path,
    )