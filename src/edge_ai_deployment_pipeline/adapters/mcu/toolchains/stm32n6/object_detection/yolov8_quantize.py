# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import logging
import warnings
import random
import tqdm
import pathlib

import tensorflow as tf
import numpy as np

from munch import DefaultMunch

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Hardcoded pre-processing values
_RESCALING_SCALE  = 255
_RESCALING_OFFSET = 0

# Hardcoded quantization I/O types
_QUANTIZATION_INPUT_TYPE  = 'uint8'
_QUANTIZATION_OUTPUT_TYPE = 'int8'


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def quantize(config: dict) -> str:
    """
    Quantize a TFLite model from a saved model.

    Parameters
    ----------
    config : dict
        {
            "model": {
                "name":        str,
                "model_path":  str,
                "input_shape": [H, W, C],
            },
            "quantization": {
                "calib_dataset_path": str,
                "export_path":        str,
                "quantization_type":  "per_channel" | "per_tensor",
            },
        }

    Returns
    -------
    str
        Full path to the generated .tflite file.
    """
    cfg = DefaultMunch.fromDict(config)

    def representative_data_gen():
        import cv2
        
        representative_ds_path = cfg.quantization.calib_dataset_path
        list_of_files = os.listdir(representative_ds_path)

        for image_file in tqdm.tqdm(list_of_files):
            if image_file.endswith(".jpg"):
                image = cv2.imread(os.path.join(representative_ds_path, image_file))
                if len(image.shape) != 3:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resized_image = cv2.resize(
                    image,
                    (int(cfg.model.input_shape[0]), int(cfg.model.input_shape[1])),
                    interpolation=cv2.INTER_LINEAR,
                )
                image_data = (
                    resized_image / _RESCALING_SCALE
                    + _RESCALING_OFFSET
                )
                img = image_data.astype(np.float32)
                yield [np.expand_dims(img, 0)]

    setup_seed(42)

    name = cfg.model.name
    print(f'Quantization of {name}')

    quant_tag  = 'quant_pc'
    input_tag  = 'f'
    output_tag = 'f'

    converter = tf.lite.TFLiteConverter.from_saved_model(cfg.model.model_path)

    tflite_models_dir = pathlib.Path(cfg.quantization.export_path)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    # Input type
    if _QUANTIZATION_INPUT_TYPE == 'int8':
        converter.inference_input_type = tf.int8
        input_tag = 'i'
    elif _QUANTIZATION_INPUT_TYPE == 'uint8':
        converter.inference_input_type = tf.uint8
        input_tag = 'u'

    # Output type
    if _QUANTIZATION_OUTPUT_TYPE == 'int8':
        converter.inference_output_type = tf.int8
        output_tag = 'i'
    elif _QUANTIZATION_OUTPUT_TYPE == 'uint8':
        converter.inference_output_type = tf.uint8
        output_tag = 'u'

    # Per-tensor vs per-channel
    if cfg.quantization.quantization_type == 'per_tensor':
        converter._experimental_disable_per_channel = True
        quant_tag = 'quant_pt'

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    tflite_model = converter.convert()
    out_filename = f'{name}_{quant_tag}_{input_tag}{output_tag}.tflite'
    out_path = str(tflite_models_dir / out_filename)
    (tflite_models_dir / out_filename).write_bytes(tflite_model)

    print(f'Quantized model generated: {out_filename}')
    return out_path