#!/usr/bin/env python3
# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  *
#  * Standalone TFLite PTQ extracted from stm32ai-modelzoo-services.
#  *
#  * This software component is provided to you as part of a software package and
#  * applicable license terms are in the Package_license file. If you received this
#  * software component outside of a package or without applicable license terms,
#  * the terms of the SLA0044 license shall apply and are fully reproduced below:
#  *
#  * SLA0044 Rev5/February 2018
#  *
#  * Software license agreement
#  *
#  * ULTIMATE LIBERTY SOFTWARE LICENSE AGREEMENT
#  *
#  * BY INSTALLING, COPYING, DOWNLOADING, ACCESSING OR OTHERWISE USING THIS SOFTWARE
#  * OR ANY PART THEREOF (AND THE RELATED DOCUMENTATION) FROM STMICROELECTRONICS
#  * INTERNATIONAL N.V, SWISS BRANCH AND/OR ITS AFFILIATED COMPANIES
#  * (STMICROELECTRONICS), THE RECIPIENT, ON BEHALF OF HIMSELF OR HERSELF, OR ON
#  * BEHALF OF ANY ENTITY BY WHICH SUCH RECIPIENT IS EMPLOYED AND/OR ENGAGED AGREES
#  * TO BE BOUND BY THIS SOFTWARE LICENSE AGREEMENT.
#  *
#  * Under STMicroelectronics' intellectual property rights, the redistribution,
#  * reproduction and use in source and binary forms of the software or any part
#  * thereof, with or without modification, are permitted provided that the following
#  * conditions are met:
#  *
#  * 1. Redistribution of source code (modified or not) must retain any copyright
#  * notice, this list of conditions and the disclaimer set forth below as items 10
#  * and 11.
#  *
#  * 2. Redistributions in binary form, except as embedded into microcontroller or
#  * microprocessor device manufactured by or for STMicroelectronics or a software
#  * update for such device, must reproduce any copyright notice provided with the
#  * binary code, this list of conditions, and the disclaimer set forth below as
#  * items 10 and 11, in documentation and/or other materials provided with the
#  * distribution.
#  *
#  * 3. Neither the name of STMicroelectronics nor the names of other contributors to
#  * this software may be used to endorse or promote products derived from this
#  * software or part thereof without specific written permission.
#  *
#  * 4. This software or any part thereof, including modifications and/or derivative
#  * works of this software, must be used and execute solely and exclusively on or in
#  * combination with a microcontroller or microprocessor device manufactured by or
#  * for STMicroelectronics.
#  *
#  * 5. No use, reproduction or redistribution of this software partially or totally
#  * may be done in any manner that would subject this software to any Open Source
#  * Terms. "Open Source Terms" shall mean any open source license which requires as
#  * part of distribution of software that the source code of such software is
#  * distributed therewith or otherwise made available, or open source license that
#  * substantially complies with the Open Source definition specified at
#  * www.opensource.org and any other comparable open source license such as for
#  * example GNU General Public License (GPL), Eclipse Public License (EPL), Apache
#  * Software License, BSD license or MIT license.
#  *
#  * 6. STMicroelectronics has no obligation to provide any maintenance, support or
#  * updates for the software.
#  *
#  * 7. The software is and will remain the exclusive property of STMicroelectronics
#  * and its licensors. The recipient will not take any action that jeopardizes
#  * STMicroelectronics and its licensors' proprietary rights or acquire any rights
#  * in the software, except the limited rights specified hereunder.
#  *
#  * 8. The recipient shall comply with all applicable laws and regulations affecting
#  * the use of the software or any part thereof including any applicable export
#  * control law or regulation.
#  *
#  * 9. Redistribution and use of this software or any part thereof other than as
#  * permitted under this license is void and will automatically terminate your
#  * rights under this license.
#  *
#  * 10. THIS SOFTWARE IS PROVIDED BY STMICROELECTRONICS AND CONTRIBUTORS "AS IS" AND
#  * ANY EXPRESS, IMPLIED OR STATUTORY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  * IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  * NON-INFRINGEMENT OF THIRD PARTY INTELLECTUAL PROPERTY RIGHTS, WHICH ARE
#  * DISCLAIMED TO THE FULLEST EXTENT PERMITTED BY LAW. IN NO EVENT SHALL
#  * STMICROELECTRONICS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
#  * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#  * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  *
#  * 11. EXCEPT AS EXPRESSLY PERMITTED HEREUNDER, NO LICENSE OR OTHER RIGHTS, WHETHER
#  * EXPRESS OR IMPLIED, ARE GRANTED UNDER ANY PATENT OR OTHER INTELLECTUAL PROPERTY
#  * RIGHTS OF STMICROELECTRONICS OR ANY THIRD PARTY.
#  *--------------------------------------------------------------------------------------------*/

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PACKAGE_ROOT = Path(__file__).resolve().parent


def ensure_package_path() -> None:
    root = str(PACKAGE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
TFLITE_NAME = "quantized_model.tflite"
OPTIMIZED_KERAS_NAME = "optimized_model.keras"
SUPPORTED_INTERPOLATIONS = [
    "nearest",
    "bilinear",
    "bicubic",
    "area",
    "lanczos3",
    "lanczos5",
    "gaussian",
    "mitchellcubic",
]


# ---------------------------------------------------------------------------
# Dataset pipeline
# ---------------------------------------------------------------------------


def get_path_dataset(
    path: str,
    class_names: List[str],
    seed: int,
    shuffle: bool = True,
):
    import tensorflow as tf

    data_list = []
    labels = []

    for d in os.scandir(path):
        if d.is_dir():
            labels.append(d.name)
        elif Path(d.name).suffix.lower() in [".jpg", ".jpeg", ".png"]:
            data_list.append((os.path.join(path, d.name), 0))

    if labels == []:
        data_list = sorted(data_list)
    else:
        data_list = []
        for label in class_names:
            if label not in labels:
                raise ValueError(f"[ERROR] label {label} not found in {path}")
        for idx, label in enumerate(sorted(class_names)):
            imgs = os.listdir(os.path.join(path, label))
            data_list.extend(sorted([(os.path.join(path, label, img), idx) for img in imgs]))

    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(data_list)

    imgs, lbls = zip(*data_list)
    return tf.data.Dataset.from_tensor_slices((list(imgs), list(lbls)))


def preprocess_function(
    data_x,
    data_y,
    image_size: Tuple[int, int],
    interpolation: str,
    aspect_ratio: str,
    color_mode: str,
    label_mode: str,
    num_classes: int,
):
    import tensorflow as tf

    height, width = image_size
    channels = 1 if color_mode == "grayscale" else 3

    image = tf.io.read_file(data_x)
    image = tf.image.decode_image(image, channels=channels, expand_animations=False)
    if aspect_ratio == "fit":
        image = tf.image.resize(
            image,
            [height, width],
            method=interpolation,
            preserve_aspect_ratio=False,
            antialias=True,
        )
    else:
        # Generic center-crop behavior
        in_shape = tf.shape(image)
        in_h = tf.cast(in_shape[0], tf.float32)
        in_w = tf.cast(in_shape[1], tf.float32)
        target_h = tf.cast(height, tf.float32)
        target_w = tf.cast(width, tf.float32)
        scale = tf.maximum(target_h / in_h, target_w / in_w)
        resized_h = tf.cast(tf.math.ceil(in_h * scale), tf.int32)
        resized_w = tf.cast(tf.math.ceil(in_w * scale), tf.int32)
        image = tf.image.resize(
            image,
            [resized_h, resized_w],
            method=interpolation,
            preserve_aspect_ratio=False,
            antialias=True,
        )
        image = tf.image.resize_with_crop_or_pad(image, height, width)

    if label_mode == "categorical":
        data_y = tf.keras.utils.to_categorical(data_y, num_classes)

    return image, data_y


def get_quantization_dataset(
    quantization_path: str,
    class_names: List[str],
    image_size: Tuple[int, int],
    interpolation: str = "nearest",
    aspect_ratio: str = "fit",
    color_mode: str = "rgb",
    seed: int = 127,
    quantization_split: float = 1.0,
):
    import tensorflow as tf

    label_mode = "int"
    batch_size = 1
    preprocess_params = (
        image_size,
        interpolation,
        aspect_ratio,
        color_mode,
        label_mode,
        len(class_names),
    )

    dataset = get_path_dataset(quantization_path, class_names, seed=seed, shuffle=False)
    dataset = dataset.map(lambda *data: preprocess_function(*data, *preprocess_params))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    if quantization_split and quantization_split != 1.0:
        logger.info("Quantizing by using %s %% of the provided dataset...", quantization_split * 100)
        dataset = dataset.take(int(len(dataset) * float(quantization_split)))

    return dataset


def preprocess_data(
    dataset,
    scale: float,
    offset: float,
    mean: float,
    std: float,
):
    import tensorflow as tf

    if dataset is None:
        return None

    rescaling = tf.keras.Sequential([tf.keras.layers.Rescaling(scale, offset)])
    dataset = dataset.map(lambda x, y: (rescaling(x), y))

    if isinstance(std, float):
        variance = pow(std, 2)
    elif isinstance(std, list):
        variance = [x ** 2 for x in std]
    else:
        variance = [x ** 2 for x in std]
    normalization = tf.keras.Sequential([tf.keras.layers.Normalization(mean=mean, variance=variance)])
    dataset = dataset.map(lambda x, y: (normalization(x), y))

    return dataset


def build_quantization_dataloader(
    calib_dir: str,
    image_size: Tuple[int, int],
    interpolation: str,
    aspect_ratio: str,
    color_mode: str,
    scale: float,
    offset: float,
    mean: float,
    std: float,
    seed: int,
    quantization_split: float,
):
    class_names = sorted(d.name for d in os.scandir(calib_dir) if d.is_dir())
    if not class_names:
        class_names = ["_default"]
    logger.info("Auto-discovered class_names: %s", class_names)

    logger.info("image_size=%s", image_size)
    quant_ds = get_quantization_dataset(
        quantization_path=calib_dir,
        class_names=class_names,
        image_size=image_size,
        interpolation=interpolation,
        aspect_ratio=aspect_ratio,
        color_mode=color_mode,
        seed=seed,
        quantization_split=quantization_split,
    )
    quant_ds = preprocess_data(quant_ds, scale=scale, offset=offset, mean=mean, std=std)
    return quant_ds

class TFLitePTQQuantizer:
    def __init__(
        self,
        model,
        quantization_ds,
        output_dir: Path,
        export_dir: str,
        quantization_input_type: str = "uint8",
        quantization_output_type: str = "float",
        granularity: str = "per_channel",
        optimize: bool = False,
    ):
        self.model = model
        self.quantization_ds = quantization_ds
        self.output_dir = output_dir
        self.export_dir = export_dir
        self.quantization_input_type = quantization_input_type
        self.quantization_output_type = quantization_output_type
        self.granularity = granularity
        self.optimize = optimize
        self.quantized_model = None

    def _representative_data_gen(self, input_shape: Tuple[int, ...]) -> Iterator[List]:
        import tqdm

        if not self.quantization_ds:
            for _ in tqdm.tqdm(range(5)):
                data = np.random.rand(1, input_shape[0], input_shape[1], input_shape[2])
                yield [data.astype(np.float32)]
        else:
            for images, _labels in tqdm.tqdm(
                self.quantization_ds, total=len(self.quantization_ds)
            ):
                for image in images:
                    import tensorflow as tf

                    image = tf.cast(image, dtype=tf.float32)
                    image = tf.expand_dims(image, 0)
                    yield [image]

    def _run_quantization(self) -> Path:
        import tensorflow as tf

        input_shape = self.model.input_shape[1:]
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        q_input = self.quantization_input_type
        q_output = self.quantization_output_type
        if q_input == "int8":
            converter.inference_input_type = tf.int8
        elif q_input == "uint8":
            converter.inference_input_type = tf.uint8
        if q_output == "int8":
            converter.inference_output_type = tf.int8
        elif q_output == "uint8":
            converter.inference_output_type = tf.uint8

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: self._representative_data_gen(input_shape)

        if self.granularity == "per_tensor":
            converter._experimental_disable_per_channel = True

        tflite_model_quantized = converter.convert()

        tflite_models_dir = pathlib.Path(os.path.join(self.output_dir, f"{self.export_dir}/"))
        tflite_models_dir.mkdir(exist_ok=True, parents=True)
        tflite_model_path = tflite_models_dir / TFLITE_NAME
        tflite_model_path.write_bytes(tflite_model_quantized)

        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        interpreter.allocate_tensors()
        setattr(interpreter, "model_path", str(tflite_model_path))
        self.quantized_model = interpreter
        return tflite_model_path

    def _prepare_quantization(self) -> None:
        import tensorflow as tf

        if not isinstance(self.model, tf.keras.Model):
            raise ValueError(f"Unsupported model format: {type(self.model)}.")

        if self.granularity == "per_tensor" and self.optimize:
            logger.info("Optimizing the model for improved per_tensor quantization...")
            ensure_package_path()
            from edge_ai_deployment_pipeline.adapters.mcu.toolchains.stm32n6.object_detection.common.optimization import model_formatting_ptq_per_tensor

            self.model = model_formatting_ptq_per_tensor(model_origin=self.model)
            models_dir = pathlib.Path(os.path.join(self.output_dir, f"{self.export_dir}/"))
            models_dir.mkdir(exist_ok=True, parents=True)
            model_path = models_dir / OPTIMIZED_KERAS_NAME
            self.model.save(model_path)

    def quantize(self) -> Path:
        logger.info("Quantizing the model ... This might take few minutes ...")
        self._prepare_quantization()
        path = self._run_quantization()
        logger.info("Quantization complete.")
        return path


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_package_path()
    parser = argparse.ArgumentParser(
        description="TensorFlow -> TFLite PTQ",
    )
    parser.add_argument("--model", required=True, help=".keras or .h5 full saved model")
    parser.add_argument(
        "--calib-dir",
        required=True,
        help="Quantization images (dataset.quantization_path); class subfolders",
    )
    parser.add_argument("--output-dir", default="./stm32n6_quantized")
    parser.add_argument(
        "--export-dir",
        default="quantized_models",
        help="Subfolder under output-dir",
    )
    parser.add_argument(
        "--interpolation",
        default="nearest",
        choices=SUPPORTED_INTERPOLATIONS,
        help="preprocessing.resizing.interpolation (quantization_config.yaml)",
    )
    parser.add_argument("--aspect-ratio", default="fit", choices=["fit", "crop"])
    parser.add_argument("--color-mode", default="rgb", choices=["rgb", "grayscale"])
    parser.add_argument("--scale", type=float, default=1.0 / 127.5)
    parser.add_argument("--offset", type=float, default=-1.0)
    parser.add_argument("--mean", type=float, default=0.0, help="preprocessing.normalization.mean")
    parser.add_argument("--std", type=float, default=1.0, help="preprocessing.normalization.std")
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--quantization-split", type=float, default=1.0)
    parser.add_argument("--granularity", default="per_channel", choices=["per_channel", "per_tensor"])
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="per_tensor only: model_formatting_ptq_per_tensor",
    )
    parser.add_argument("--input-type", default="uint8", choices=["float", "int8", "uint8"])
    parser.add_argument("--output-type", default="float", choices=["float", "int8", "uint8"])
    parser.add_argument(
        "--image-size",
        default=None,
        help="H,W override (default: from model input, like prepare_kwargs_for_dataloader)",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    calib_dir = Path(args.calib_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not model_path.is_file() or model_path.suffix.lower() not in (".keras", ".h5"):
        logger.error("--model must be a .keras or .h5 file")
        return 1
    if not calib_dir.is_dir():
        logger.error("Calibration dir not found: %s", calib_dir)
        return 1
    if args.optimize and args.granularity != "per_tensor":
        logger.error("--optimize only applies with --granularity per_tensor")
        return 1

    try:
        import tensorflow as tf  # noqa: F401

        model = tf.keras.models.load_model(str(model_path), compile=False)
        if not isinstance(model, tf.keras.Model):
            raise TypeError(f"Expected tf.keras.Model, got {type(model)}")
        logger.info("Loaded Keras model: %s", model_path)
        logger.info("input_shape=%s", model.inputs[0].shape[1:])
        if args.image_size:
            parts = [int(x.strip()) for x in args.image_size.replace(" ", "").split(",")]
            image_size = (parts[0], parts[1])
        else:
            # Same as prepare_kwargs_for_dataloader
            image_size = tuple(model.inputs[0].shape[1:])[:-1]

        quant_ds = build_quantization_dataloader(
            calib_dir=str(calib_dir),
            image_size=image_size,
            interpolation=args.interpolation,
            aspect_ratio=args.aspect_ratio,
            color_mode=args.color_mode,
            scale=args.scale,
            offset=args.offset,
            mean=args.mean,
            std=args.std,
            seed=args.seed,
            quantization_split=args.quantization_split,
        )

        quantizer = TFLitePTQQuantizer(
            model=model,
            quantization_ds=quant_ds,
            output_dir=output_dir,
            export_dir=args.export_dir,
            quantization_input_type=args.input_type,
            quantization_output_type=args.output_type,
            granularity=args.granularity,
            optimize=args.optimize,
        )
        tflite_path = quantizer.quantize()
        flat = output_dir / TFLITE_NAME
        if tflite_path != flat:
            flat.write_bytes(tflite_path.read_bytes())

        logger.info("Done.")
        logger.info("  TFLite: %s", tflite_path)
        logger.info("  Copy:   %s", flat)
        return 0
    except Exception as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
