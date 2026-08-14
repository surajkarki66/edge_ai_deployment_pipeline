#!/usr/bin/env python3
import argparse
import os
import pickle
import time
import cv2
import numpy as np
import vart
import xir

from typing import Tuple


# FPGA / VART helpers
def get_child_subgraph_dpu(graph: "xir.Graph"):
    root_subgraph = graph.get_root_subgraph()
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def classify_preprocess(image, target_width, target_height, input_scale,
                         resize_short_side=None, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
    """
    Standard classification preprocessing: resize -> center crop -> normalize -> quantize.

    Args:
        image: BGR image from cv2
        target_width/target_height: model input spatial size (after crop)
        input_scale: 2 ** fix_point from the DPU input tensor
        resize_short_side: if set, resize so the shorter side equals this value before
                            center-cropping to (target_width, target_height). If None,
                            resize_short_side defaults to target size * 1.14 (common ImageNet-style ratio).
        mean/std: normalization stats applied on the [0,1]-scaled RGB image before quantization

    Returns:
        processed: np.int8 array ready for DPU, shape (target_height, target_width, 3)
        meta: dict with transform parameters (kept for parity with detection script / debugging)
    """
    h0, w0 = image.shape[:2]

    if resize_short_side is None:
        resize_short_side = int(round(min(target_width, target_height) * 1.14))

    # Resize preserving aspect ratio so the shorter side == resize_short_side
    if h0 < w0:
        new_h = resize_short_side
        new_w = int(round(w0 * (resize_short_side / h0)))
    else:
        new_w = resize_short_side
        new_h = int(round(h0 * (resize_short_side / w0)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop to (target_width, target_height)
    top = max(0, (new_h - target_height) // 2)
    left = max(0, (new_w - target_width) // 2)
    cropped = resized[top: top + target_height, left: left + target_width]

    # Guard against off-by-one rounding leaving the crop short of target size
    ch, cw = cropped.shape[:2]
    if ch != target_height or cw != target_width:
        cropped = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # BGR -> RGB, normalize, quantize
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    normed = (rgb - mean_arr) / std_arr
    quantized = (normed * input_scale).astype(np.int8)

    meta = {
        "mode": "classify_center_crop",
        "h0": h0,
        "w0": w0,
        "resize_short_side": resize_short_side,
        "target_w": target_width,
        "target_h": target_height,
    }
    return quantized, meta


def softmax_1d(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def topk(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


# Single-image inference
def run_image_classify(
    model_path: str,
    config_path: str,
    image_path: str,
    output_dir: str,
    top_k: int,
):
    os.makedirs(output_dir, exist_ok=True)

    g = xir.Graph.deserialize(model_path)
    subgraphs = get_child_subgraph_dpu(g)
    if not subgraphs:
        raise RuntimeError("No DPU subgraph found in the xmodel.")
    runner = vart.Runner.create_runner(subgraphs[0], "run")

    input_t = runner.get_input_tensors()[0]
    in_shape = tuple(input_t.dims)  # expected NHWC or NCHW
    if len(in_shape) != 4:
        raise ValueError(f"Unexpected input tensor dims: {in_shape}")
    in_scale = 2 ** input_t.get_attr("fix_point")

    out_tensors = runner.get_output_tensors()
    if len(out_tensors) != 1:
        raise ValueError(
            f"Classification script expects a single logits/prob output tensor, got {len(out_tensors)}"
        )
    out_fixpoint = out_tensors[0].get_attr("fix_point")
    out_scale = float(2 ** out_fixpoint)

    # config.pkl for classification: (class_names, needs_softmax, mean, std)
    # class_names: list[str]; needs_softmax: bool (False if output is already probabilities,
    # e.g. softmax fused into the graph); mean/std: normalization stats used at training time.
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    class_names = cfg[0]
    needs_softmax = bool(cfg[1]) if len(cfg) > 1 else True
    mean = tuple(cfg[2]) if len(cfg) > 2 else (0.0, 0.0, 0.0)
    std = tuple(cfg[3]) if len(cfg) > 3 else (1.0, 1.0, 1.0)

    input_is_nhwc = in_shape[-1] == 3
    input_is_nchw = in_shape[1] == 3 and not input_is_nhwc
    if not (input_is_nhwc or input_is_nchw):
        raise ValueError(f"Unsupported input layout for dims {in_shape}. Expected RGB on channel axis.")

    if int(in_shape[0]) != 1:
        raise ValueError(f"This script currently supports batch size 1, got input batch {in_shape[0]}")

    in_h = int(in_shape[1]) if input_is_nhwc else int(in_shape[2])
    in_w = int(in_shape[2]) if input_is_nhwc else int(in_shape[3])

    out_shape = tuple(out_tensors[0].dims)
    if int(out_shape[0]) != 1:
        raise ValueError(f"This script currently supports batch size 1, got output batch {out_shape[0]}")
    num_classes = int(np.prod(out_shape[1:]))
    if class_names and len(class_names) != num_classes:
        print(
            f"WARNING: config has {len(class_names)} class names but model outputs "
            f"{num_classes} logits; falling back to numeric class ids where names are missing."
        )

    print(f"Input shape: {in_shape}, target crop={in_w}x{in_h}")
    print(f"Preprocessing: resize-short-side + center crop, mean={mean}, std={std}")
    print(f"Classifier params: num_classes={num_classes}, needs_softmax={needs_softmax}, top_k={top_k}")

    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    in_data = [np.empty(in_shape, dtype=np.int8, order="C")]
    out_data = [np.empty(out_shape, dtype=np.int8, order="C")]

    end_to_end_start = time.perf_counter()

    processed_img, meta = classify_preprocess(
        frame, in_w, in_h, in_scale, mean=mean, std=std
    )

    if input_is_nhwc:
        in_data[0][0, ...] = processed_img
    else:
        in_data[0][0, 0, :, :] = processed_img[:, :, 0]
        in_data[0][0, 1, :, :] = processed_img[:, :, 1]
        in_data[0][0, 2, :, :] = processed_img[:, :, 2]

    fpga_start = time.perf_counter()
    jid = runner.execute_async(in_data, out_data)
    runner.wait(jid)
    fpga_end = time.perf_counter()

    logits = (out_data[0].astype(np.float32) / out_scale).reshape(-1)

    if needs_softmax:
        probs = softmax_1d(logits)
    else:
        probs = logits

    top_idx, top_scores = topk(probs, top_k)

    end_to_end_end = time.perf_counter()

    f_lat = (fpga_end - fpga_start) * 1000.0
    end_to_end_lat = (end_to_end_end - end_to_end_start) * 1000.0

    print("-" * 100)
    print(f"FPGA latency:      {f_lat:.2f} ms ({1000.0/f_lat:.2f} FPS)")
    print(f"End-to-end latency: {end_to_end_lat:.2f} ms ({1000.0/end_to_end_lat:.2f} FPS)")
    print(f"Overhead (non-FPGA): {end_to_end_lat - f_lat:.2f} ms")
    print("-" * 100)

    print(f"Top-{top_k} predictions:")
    for rank, (cls_i, score) in enumerate(zip(top_idx, top_scores)):
        if class_names and cls_i < len(class_names):
            label = f"{class_names[cls_i]}: {score:.4f}"
        else:
            label = f"class_{int(cls_i)}: {score:.4f}"
        print(f"  {rank+1}. {label}")

    # Overlay top-k predictions on the frame
    for rank, (cls_i, score) in enumerate(zip(top_idx, top_scores)):
        if class_names and cls_i < len(class_names):
            label = f"{class_names[cls_i]}: {score:.2f}"
        else:
            label = f"class_{int(cls_i)}: {score:.2f}"
        cv2.putText(
            frame,
            label,
            (10, 25 + rank * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
    cv2.imwrite(out_path, frame)
    print(f"Annotated image saved to: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Single-image FPGA inference for an image classification model using VART."
    )
    p.add_argument("model_path", help="Path to .xmodel")
    p.add_argument(
        "config_path",
        help="Path to config.pkl containing (class_names, needs_softmax, mean, std)",
    )
    p.add_argument("image_path", help="Path to input image file")
    p.add_argument("output_dir", help="Output directory for the annotated image")
    p.add_argument("--top_k", type=int, default=5, help="Number of top predictions to display")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_image_classify(
        model_path=args.model_path,
        config_path=args.config_path,
        image_path=args.image_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )