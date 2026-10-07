from __future__ import annotations

import argparse
import csv
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

# MNIST normalization constants used by transforms.Normalize((0.1307,), (0.3081,))
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


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
# 2. Allocate device/host buffers for all bindings (once, reused every image)
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
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        shape = context.get_tensor_shape(name)
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
        context.set_tensor_address(name, int(device_mem))

    return bindings, stream


# --------------------------------------------------------------------------
# 3. Preprocess: matches the MNIST test_transform pipeline
# --------------------------------------------------------------------------
def preprocess_image(image_path: str, input_shape: tuple[int, int, int, int]) -> np.ndarray:
    """Resize + normalize a grayscale image to match:

        transforms.ToTensor()                      # scales pixels to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # (x - mean) / std
    """
    import cv2

    n, c, h, w = input_shape
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32) / 255.0
    img = (img - MNIST_MEAN) / MNIST_STD
    img = np.expand_dims(img, axis=0)  # channel dim
    img = np.expand_dims(img, axis=0)  # batch dim
    return np.ascontiguousarray(img)


# --------------------------------------------------------------------------
# 4. Single inference call, reusing pre-allocated bindings/context/stream
# --------------------------------------------------------------------------
def run_inference(bindings: list[Binding], stream, context: trt.IExecutionContext,
                   input_array: np.ndarray) -> np.ndarray:
    for b in bindings:
        if b.is_input:
            np.copyto(b.host, input_array.ravel())
            cuda.memcpy_htod_async(b.device, b.host, stream)

    context.execute_async_v3(stream_handle=stream.handle)

    output = None
    for b in bindings:
        if not b.is_input:
            cuda.memcpy_dtoh_async(b.host, b.device, stream)

    stream.synchronize()

    for b in bindings:
        if not b.is_input:
            output = np.array(b.host).reshape(b.shape)

    return output


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# --------------------------------------------------------------------------
# 5. Main evaluation loop
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="Path to model.engine")
    parser.add_argument("--labels", required=True, help="Path to labels.csv (filename,label)")
    parser.add_argument("--images_dir", required=True, help="Directory containing the test images")
    parser.add_argument("--output_json", default="eval_results.json",
                         help="Path to save evaluation results as JSON (default: eval_results.json)")
    parser.add_argument("--show_errors", type=int, default=10,
                         help="Print up to N misclassified examples")
    args = parser.parse_args()

    print(f"Loading engine: {args.engine}")
    metadata, engine = load_trt_engine_with_metadata(args.engine)
    print(f"Embedded metadata: {json.dumps(metadata, indent=2)}")

    context = engine.create_execution_context()

    # Determine input shape from the engine
    input_shape = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_shape = tuple(engine.get_tensor_shape(name))
            if -1 in input_shape:
                # Fall back to a concrete shape and set it explicitly
                meta_shape = metadata.get("input_shape")
                concrete = (1, *meta_shape) if meta_shape and len(meta_shape) == 3 else (1, 1, 28, 28)
                context.set_input_shape(name, concrete)
                input_shape = concrete
            break
    if input_shape is None:
        input_shape = (1, 1, 28, 28)
    print(f"Input shape: {input_shape}")

    bindings, stream = allocate_buffers(engine, context)

    images_dir = Path(args.images_dir)

    # Read labels.csv
    rows = []
    with open(args.labels, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["filename"], int(row["label"])))

    print(f"Evaluating {len(rows)} images from {images_dir} ...")

    correct = 0
    total = 0
    errors = []
    predictions = []
    num_classes = None
    confusion = None

    start = time.time()
    for filename, true_label in rows:
        image_path = images_dir / (Path(filename).stem + ".jpg")
        if not image_path.exists():
            print(f"  [warn] missing file, skipping: {image_path}")
            continue

        input_array = preprocess_image(str(image_path), input_shape)
        logits = run_inference(bindings, stream, context, input_array)
        logits = logits.reshape(-1)

        if num_classes is None:
            num_classes = logits.shape[0]
            confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

        probs = softmax(logits)
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label])

        confusion[true_label, pred_label] += 1
        total += 1
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        elif len(errors) < args.show_errors:
            errors.append((filename, true_label, pred_label, confidence))

        predictions.append({
            "filename": filename,
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": confidence,
            "correct": is_correct,
        })

    elapsed = time.time() - start

    print(f"\nEvaluated {total} images in {elapsed:.2f} s "
          f"({elapsed / max(total, 1) * 1000:.2f} ms/image)")
    accuracy = correct / total if total else 0.0
    print(f"Accuracy: {correct}/{total} = {accuracy:.4%}")

    if errors:
        print(f"\nSample misclassifications (up to {args.show_errors}):")
        for filename, true_label, pred_label, confidence in errors:
            print(f"  {filename}: true={true_label} pred={pred_label} (confidence={confidence:.4f})")

    if confusion is not None:
        print("\nConfusion matrix (rows=true, cols=predicted):")
        header = "      " + " ".join(f"{c:>5}" for c in range(num_classes))
        print(header)
        for i in range(num_classes):
            row_str = " ".join(f"{confusion[i, j]:>5}" for j in range(num_classes))
            print(f"true {i}: {row_str}")

        # Per-class accuracy
        print("\nPer-class accuracy:")
        per_class_accuracy = {}
        for i in range(num_classes):
            class_total = int(confusion[i].sum())
            class_correct = int(confusion[i, i])
            class_acc = class_correct / class_total if class_total else 0.0
            per_class_accuracy[str(i)] = {
                "correct": class_correct,
                "total": class_total,
                "accuracy": class_acc,
            }
            print(f"  class {i}: {class_correct}/{class_total} = {class_acc:.4%}")
    else:
        per_class_accuracy = {}

    results = {
        "engine": args.engine,
        "labels_csv": args.labels,
        "images_dir": str(images_dir),
        "metadata": metadata,
        "total_images": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "ms_per_image": elapsed / max(total, 1) * 1000,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist() if confusion is not None else [],
        "predictions": predictions,
    }

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
