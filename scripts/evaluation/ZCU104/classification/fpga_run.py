#!/usr/bin/env python3

import argparse
import glob
import os
import pickle
import time

import cv2
import numpy as np
import vart
import xir


# --------------------------------------------------------------------------
# Preprocessing / math helpers
# --------------------------------------------------------------------------
def classify_preprocess(image, target_width, target_height, input_scale,
                         resize_short_side=None, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
                         grayscale=False):
    """
    Resize -> center crop -> normalize -> quantize.
    If grayscale=True, treats the image as single-channel (MNIST-style) and
    skips the BGR->RGB conversion; mean/std are expected to be length-1 tuples.
    """
    if grayscale and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h0, w0 = image.shape[:2]

    if resize_short_side is None:
        resize_short_side = int(round(min(target_width, target_height) * 1.14))

    if h0 < w0:
        new_h = resize_short_side
        new_w = int(round(w0 * (resize_short_side / h0)))
    else:
        new_w = resize_short_side
        new_h = int(round(h0 * (resize_short_side / w0)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = max(0, (new_h - target_height) // 2)
    left = max(0, (new_w - target_width) // 2)
    cropped = resized[top: top + target_height, left: left + target_width]

    ch, cw = cropped.shape[:2]
    if ch != target_height or cw != target_width:
        cropped = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    if grayscale:
        arr = cropped.astype(np.float32) / 255.0
        arr = arr.reshape(target_height, target_width, 1)
        mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        std_arr = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    else:
        arr = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    normed = (arr - mean_arr) / std_arr
    quantized = (normed * input_scale).astype(np.int8)

    meta = {
        "mode": "classify_center_crop",
        "h0": h0, "w0": w0,
        "resize_short_side": resize_short_side,
        "target_w": target_width, "target_h": target_height,
        "grayscale": grayscale,
    }
    return quantized, meta


def softmax_1d(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def get_child_subgraph_dpu(graph):
    root_subgraph = graph.get_root_subgraph()
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


# --------------------------------------------------------------------------
# Test-set discovery
# --------------------------------------------------------------------------
def build_test_list(test_dir: str):
    """
    Returns list of (image_path, true_label:int|None) from a class-subfolder layout:
        test_dir/<digit_label>/*.png
    """
    samples = []

    subdirs = [d for d in sorted(os.listdir(test_dir))
               if os.path.isdir(os.path.join(test_dir, d))]
    if not subdirs:
        raise ValueError(
            f"No class subfolders found in {test_dir}. "
            "Expected layout: test_dir/<digit_label>/*.png"
        )

    for d in subdirs:
        try:
            label = int(d)
        except ValueError:
            label = d  # keep string label as-is if not numeric
        for img_path in sorted(glob.glob(os.path.join(test_dir, d, "*"))):
            if os.path.splitext(img_path)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                samples.append((img_path, label))

    return samples


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
def run_predict(model_path, config_path, test_dir, output_pkl, grayscale=False):
    g = xir.Graph.deserialize(model_path)
    subgraphs = get_child_subgraph_dpu(g)
    if not subgraphs:
        raise RuntimeError("No DPU subgraph found in the xmodel.")
    runner = vart.Runner.create_runner(subgraphs[0], "run")

    input_t = runner.get_input_tensors()[0]
    in_shape = tuple(input_t.dims)
    if len(in_shape) != 4:
        raise ValueError(f"Unexpected input tensor dims: {in_shape}")
    in_scale = 2 ** input_t.get_attr("fix_point")

    out_tensors = runner.get_output_tensors()
    if len(out_tensors) != 1:
        raise ValueError(f"Expected a single logits/prob output tensor, got {len(out_tensors)}")
    out_fixpoint = out_tensors[0].get_attr("fix_point")
    out_scale = float(2 ** out_fixpoint)

    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    class_names = cfg[0]
    needs_softmax = bool(cfg[1]) if len(cfg) > 1 else True
    n_channels = 1 if grayscale else 3
    mean = tuple(cfg[2]) if len(cfg) > 2 else tuple([0.0] * n_channels)
    std = tuple(cfg[3]) if len(cfg) > 3 else tuple([1.0] * n_channels)

    input_is_nhwc = in_shape[-1] in (1, 3)
    input_is_nchw = in_shape[1] in (1, 3) and not input_is_nhwc
    if not (input_is_nhwc or input_is_nchw):
        raise ValueError(f"Unsupported input layout for dims {in_shape}.")
    if int(in_shape[0]) != 1:
        raise ValueError(f"This script supports batch size 1, got input batch {in_shape[0]}")

    in_h = int(in_shape[1]) if input_is_nhwc else int(in_shape[2])
    in_w = int(in_shape[2]) if input_is_nhwc else int(in_shape[3])

    out_shape = tuple(out_tensors[0].dims)
    if int(out_shape[0]) != 1:
        raise ValueError(f"This script supports batch size 1, got output batch {out_shape[0]}")
    num_classes = int(np.prod(out_shape[1:]))
    if class_names and len(class_names) != num_classes:
        print(f"WARNING: config has {len(class_names)} class names but model outputs "
              f"{num_classes} logits.")

    samples = build_test_list(test_dir)
    if not samples:
        raise RuntimeError(f"No test images found under {test_dir}")
    print(f"Found {len(samples)} test images.")
    print(f"Input shape: {in_shape}, target crop={in_w}x{in_h}, grayscale={grayscale}")

    results = []
    in_data = [np.empty(in_shape, dtype=np.int8, order="C")]
    out_data = [np.empty(out_shape, dtype=np.int8, order="C")]

    overall_start = time.perf_counter()
    for i, (img_path, true_label) in enumerate(samples):
        frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if frame is None:
            print(f"WARNING: could not read {img_path}, skipping.")
            continue

        processed_img, _ = classify_preprocess(
            frame, in_w, in_h, in_scale, mean=mean, std=std, grayscale=grayscale
        )

        if input_is_nhwc:
            if grayscale:
                in_data[0][0, :, :, 0] = processed_img[:, :, 0]
            else:
                in_data[0][0, ...] = processed_img
        else:
            for c in range(n_channels):
                in_data[0][0, c, :, :] = processed_img[:, :, c]

        fpga_start = time.perf_counter()
        jid = runner.execute_async(in_data, out_data)
        runner.wait(jid)
        fpga_end = time.perf_counter()

        logits = (out_data[0].astype(np.float32) / out_scale).reshape(-1).copy()
        probs = softmax_1d(logits) if needs_softmax else logits
        pred_label = int(np.argmax(probs))

        results.append({
            "image_path": img_path,
            "true_label": true_label,
            "pred_label": pred_label,
            "pred_score": float(probs[pred_label]),
            "all_probs": probs.copy(),
            "latency_ms": (fpga_end - fpga_start) * 1000.0,
        })

        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(samples)}")

    overall_end = time.perf_counter()
    total_s = overall_end - overall_start

    payload = {
        "class_names": class_names,
        "num_classes": num_classes,
        "num_samples": len(results),
        "total_time_s": total_s,
        "results": results,
    }
    with open(output_pkl, "wb") as f:
        pickle.dump(payload, f)

    print("-" * 80)
    print(f"Ran {len(results)} samples in {total_s:.2f}s "
          f"({len(results)/total_s:.2f} samples/s overall)")
    print(f"Predictions saved to: {output_pkl}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run FPGA/DPU inference over a test set and dump raw predictions."
    )
    p.add_argument("model_path", help="Path to .xmodel")
    p.add_argument("config_path", help="Path to config.pkl containing (class_names, needs_softmax, mean, std)")
    p.add_argument("test_dir", help="Path to test set (class-subfolder layout: test_dir/<label>/*.png)")
    p.add_argument("output_pkl", help="Where to save raw predictions (pickle)")
    p.add_argument("--grayscale", action="store_true", help="Treat input as single-channel (MNIST)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_predict(
        model_path=args.model_path,
        config_path=args.config_path,
        test_dir=args.test_dir,
        output_pkl=args.output_pkl,
        grayscale=args.grayscale,
    )