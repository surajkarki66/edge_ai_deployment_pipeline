from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
except ImportError as e:
    raise SystemExit(
        "pycuda is required. Install with: pip install pycuda --break-system-packages"
    ) from e


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# --------------------------------------------------------------------------
# 1. Load engine (strip metadata header first)
# --------------------------------------------------------------------------
def load_trt_engine_with_metadata(path: str | Path) -> tuple[dict, trt.ICudaEngine]:
    path = Path(path)
    with open(path, "rb") as f:
        meta_len = int.from_bytes(f.read(4), byteorder="little", signed=True)
        metadata_bytes = f.read(meta_len)
        engine_bytes = f.read()  # remainder is the real TRT engine

    try:
        metadata = json.loads(metadata_bytes.decode())
    except Exception:
        raise ValueError(
            f"Could not parse metadata header from {path}. "
            "This file may not have been exported with metadata=..., "
            "in which case just call runtime.deserialize_cuda_engine() "
            "directly on the raw file bytes instead."
        )

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine after stripping metadata.")

    return metadata, engine


# --------------------------------------------------------------------------
# 2. Allocate device/host buffers for all bindings
# --------------------------------------------------------------------------
class Binding:
    def __init__(self, name, dtype, shape, host_mem, device_mem, is_input):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.host = host_mem
        self.device = device_mem
        self.is_input = is_input


def allocate_buffers(engine: trt.ICudaEngine, context: trt.IExecutionContext):
    bindings = []
    device_ptrs = []
    stream = cuda.Stream()

    num_io = engine.num_io_tensors
    for i in range(num_io):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        shape = context.get_tensor_shape(name)  # resolved shape (after any set_input_shape)
        dtype = trt.nptype(engine.get_tensor_dtype(name))

        size = int(np.prod(shape)) if -1 not in shape else 0
        if size <= 0:
            raise RuntimeError(
                f"Tensor '{name}' has dynamic/unset shape {tuple(shape)}. "
                "Call context.set_input_shape(name, concrete_shape) before allocating buffers."
            )

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(
            Binding(name=name, dtype=dtype, shape=tuple(shape),
                    host_mem=host_mem, device_mem=device_mem, is_input=is_input)
        )
        device_ptrs.append(int(device_mem))

        # Tell the context which device address this tensor uses
        context.set_tensor_address(name, int(device_mem))

    return bindings, stream


# --------------------------------------------------------------------------
# 3. Preprocess: grayscale 28x28
# --------------------------------------------------------------------------
# MNIST normalization constants used by transforms.Normalize((0.1307,), (0.3081,))
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def preprocess_image(image_path: str, input_shape: tuple[int, int, int, int]) -> np.ndarray:
    """Resize + normalize a grayscale image to match the MNIST training pipeline:

        transforms.ToTensor()                      # scales pixels to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # (x - mean) / std

    Matches a (1, 1, H, W) model input.
    """
    import cv2

    n, c, h, w = input_shape
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # read as 1-channel
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32) / 255.0  # equivalent to ToTensor()'s scaling
    img = (img - MNIST_MEAN) / MNIST_STD  # equivalent to Normalize((0.1307,), (0.3081,))
    img = np.expand_dims(img, axis=0)  # add channel dim -> (1, H, W)
    img = np.expand_dims(img, axis=0)  # add batch dim   -> (1, 1, H, W)
    return np.ascontiguousarray(img)


# --------------------------------------------------------------------------
# 4. Postprocess: logits -> probabilities -> predicted class
# --------------------------------------------------------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def decode_prediction(logits: np.ndarray) -> tuple[int, float]:
    """Return (predicted_class, confidence) from a (1, num_classes) or (num_classes,) logits array."""
    logits = logits.reshape(-1)
    probs = softmax(logits)
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    return predicted_class, confidence


# --------------------------------------------------------------------------
# 5. Run inference
# --------------------------------------------------------------------------
def run_inference(engine: trt.ICudaEngine, input_array: np.ndarray) -> dict[str, np.ndarray]:
    context = engine.create_execution_context()

    # If the engine has a dynamic input shape, set it explicitly before allocating
    input_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
            if -1 in engine.get_tensor_shape(name):
                context.set_input_shape(name, input_array.shape)
            break

    bindings, stream = allocate_buffers(engine, context)

    # Copy input data into the pagelocked host buffer for the input tensor
    for b in bindings:
        if b.is_input:
            np.copyto(b.host, input_array.ravel())
            cuda.memcpy_htod_async(b.device, b.host, stream)

    # Run inference (async, TensorRT 10.x API)
    context.execute_async_v3(stream_handle=stream.handle)

    # Copy outputs back to host
    for b in bindings:
        if not b.is_input:
            cuda.memcpy_dtoh_async(b.host, b.device, stream)

    stream.synchronize()

    outputs = {}
    for b in bindings:
        if not b.is_input:
            outputs[b.name] = np.array(b.host).reshape(b.shape)

    return outputs


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="Path to model.engine")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    args = parser.parse_args()

    print(f"Loading engine: {args.engine}")
    metadata, engine = load_trt_engine_with_metadata(args.engine)
    print(f"Embedded metadata: {json.dumps(metadata, indent=2)}")

    # Determine input shape from the engine itself
    input_shape = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_shape = tuple(engine.get_tensor_shape(name))
            break
    if input_shape is None or -1 in input_shape:
        # Fall back to metadata's recorded input_shape, else a default.
        meta_shape = metadata.get("input_shape")
        if meta_shape and len(meta_shape) == 3:
            input_shape = (1, *meta_shape)  # (C, H, W) -> (1, C, H, W)
        else:
            input_shape = (1, 1, 28, 28)
    print(f"Input shape: {input_shape}")

    input_array = preprocess_image(args.input, input_shape)

    # Warmup (first inference on Jetson includes context/cuda-graph setup cost)
    for _ in range(args.warmup):
        run_inference(engine, input_array)

    start = time.time()
    outputs = run_inference(engine, input_array)
    elapsed = (time.time() - start) * 1000

    print(f"\nInference time: {elapsed:.2f} ms")
    for name, arr in outputs.items():
        print(f"Output '{name}': shape={arr.shape}, dtype={arr.dtype}")
        print(f"  sample values: {arr.ravel()[:10]}")
        predicted_class, confidence = decode_prediction(arr)
        print(f"  Predicted class: {predicted_class} (confidence: {confidence:.4f})")


if __name__ == "__main__":
    main()
