#!/usr/bin/env python3

import argparse
import csv
import os
import pickle
import time

import cv2
import numpy as np
import vart
import xir


# ============================================================================
# DPU / XIR helpers
# ============================================================================

def get_child_subgraph_dpu(graph):
    root_subgraph = graph.get_root_subgraph()
    child_subgraphs = root_subgraph.toposort_child_subgraph()

    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device")
        and cs.get_attr("device").upper() == "DPU"
    ]


# ============================================================================
# CSV
# ============================================================================

def load_labels(csv_path):
    """
    CSV format:

        filename,label
        821.png,0
        5942.png,0
        312.png,0
        6978.png,0
    """

    labels = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        if "filename" not in reader.fieldnames:
            raise ValueError(
                "CSV must contain a 'filename' column."
            )

        if "label" not in reader.fieldnames:
            raise ValueError(
                "CSV must contain a 'label' column."
            )

        for row in reader:
            filename = row["filename"].strip()
            label = int(row["label"])

            labels[filename] = label

    return labels


# ============================================================================
# MNIST preprocessing
# ============================================================================

def preprocess_mnist(image, input_scale):
    """
    Exactly matches the training preprocessing:

        ToTensor()
        Normalize((0.1307,), (0.3081,))

    Training:

        x = pixel / 255
        x = (x - 0.1307) / 0.3081

    Then quantization:

        x_quant = x * input_scale

    """

    if image is None:
        raise ValueError("Input image is None.")

    # Convert to grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # MNIST is 28x28
    if image.shape != (28, 28):
        image = cv2.resize(
            image,
            (28, 28),
            interpolation=cv2.INTER_LINEAR
        )

    # Same as torchvision.transforms.ToTensor()
    image = image.astype(np.float32) / 255.0

    # Same as:
    # transforms.Normalize((0.1307,), (0.3081,))
    image = (image - 0.1307) / 0.3081

    # Quantize for DPU
    image = image * input_scale

    image = image.astype(np.int8)

    # NHWC:
    # (28, 28) -> (28, 28, 1)
    image = image.reshape(28, 28, 1)

    return image


# ============================================================================
# FPGA inference
# ============================================================================

def run_fpga_test(
    model_path,
    image_dir,
    csv_path,
    output_pkl,
):

    print("=" * 80)
    print("Loading XModel:", model_path)
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Load XModel
    # ------------------------------------------------------------------------

    graph = xir.Graph.deserialize(model_path)

    subgraphs = get_child_subgraph_dpu(graph)

    if not subgraphs:
        raise RuntimeError(
            "No DPU subgraph found in the XModel."
        )

    print("DPU subgraph:", subgraphs[0].get_name())

    runner = vart.Runner.create_runner(
        subgraphs[0],
        "run"
    )

    # ------------------------------------------------------------------------
    # Input tensor
    # ------------------------------------------------------------------------

    input_tensor = runner.get_input_tensors()[0]

    input_shape = tuple(input_tensor.dims)

    input_fix_point = input_tensor.get_attr("fix_point")

    input_scale = 2 ** input_fix_point

    # ------------------------------------------------------------------------
    # Output tensor
    # ------------------------------------------------------------------------

    output_tensors = runner.get_output_tensors()

    if len(output_tensors) != 1:
        raise RuntimeError(
            f"Expected 1 output tensor, got {len(output_tensors)}"
        )

    output_tensor = output_tensors[0]

    output_shape = tuple(output_tensor.dims)

    output_fix_point = output_tensor.get_attr("fix_point")

    output_scale = 2 ** output_fix_point

    # ------------------------------------------------------------------------
    # Print model information
    # ------------------------------------------------------------------------

    print()
    print("MODEL INFORMATION")
    print("-" * 80)

    print("Input shape:       ", input_shape)
    print("Input fix point:   ", input_fix_point)
    print("Input scale:       ", input_scale)

    print("Output shape:      ", output_shape)
    print("Output fix point:  ", output_fix_point)
    print("Output scale:      ", output_scale)

    # We expect MNIST
    if input_shape != (1, 28, 28, 1):
        raise RuntimeError(
            f"Expected MNIST input shape (1,28,28,1), "
            f"but got {input_shape}"
        )

    if output_shape != (1, 10):
        raise RuntimeError(
            f"Expected MNIST output shape (1,10), "
            f"but got {output_shape}"
        )

    # ------------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------------

    labels = load_labels(csv_path)

    print()
    print(f"Labels loaded from CSV: {len(labels)}")

    # ------------------------------------------------------------------------
    # Find images
    # ------------------------------------------------------------------------

    valid_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
    }

    image_files = []

    for filename in os.listdir(image_dir):

        extension = os.path.splitext(filename)[1].lower()

        if extension not in valid_extensions:
            continue

        image_path = os.path.join(
            image_dir,
            filename
        )

        if filename not in labels:
            print(
                f"WARNING: {filename} not found in CSV. Skipping."
            )
            continue

        image_files.append(
            (filename, image_path, labels[filename])
        )

    image_files.sort()

    if not image_files:
        raise RuntimeError(
            "No images found that have corresponding labels in CSV."
        )

    print(f"Images to test: {len(image_files)}")

    # ------------------------------------------------------------------------
    # Allocate DPU buffers
    # ------------------------------------------------------------------------

    input_data = [
        np.empty(
            input_shape,
            dtype=np.int8,
            order="C"
        )
    ]

    output_data = [
        np.empty(
            output_shape,
            dtype=np.int8,
            order="C"
        )
    ]

    # ------------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------------

    results = []

    total_fpga_time = 0.0

    successful = 0

    print()
    print("=" * 80)
    print("STARTING FPGA INFERENCE")
    print("=" * 80)

    for index, (filename, image_path, true_label) in enumerate(image_files):

        # ------------------------------------------------------------
        # Read image
        # ------------------------------------------------------------

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(
                f"WARNING: Could not read {image_path}"
            )
            continue

        # ------------------------------------------------------------
        # Preprocessing
        # ------------------------------------------------------------

        processed = preprocess_mnist(
            image,
            input_scale
        )

        # NHWC
        input_data[0][0, :, :, 0] = processed[:, :, 0]

        # ------------------------------------------------------------
        # FPGA timing
        # ------------------------------------------------------------

        start = time.perf_counter()

        job_id = runner.execute_async(
            input_data,
            output_data
        )

        runner.wait(job_id)

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000.0

        total_fpga_time += (
            end - start
        )

        # ------------------------------------------------------------
        # Get raw output
        # ------------------------------------------------------------

        raw_output = (
            output_data[0]
            .reshape(-1)
            .copy()
        )

        # ------------------------------------------------------------
        # Dequantize
        # ------------------------------------------------------------

        scores = (
            raw_output.astype(np.float32)
            / output_scale
        )
        pred_label = int(
            np.argmax(scores)
        )

        pred_score = float(
            scores[pred_label]
        )

        # ------------------------------------------------------------
        # Store result
        # ------------------------------------------------------------

        results.append({
            "filename": filename,
            "image_path": image_path,
            "true_label": int(true_label),
            "pred_label": pred_label,
            "pred_score": pred_score,
            "raw_output": raw_output,
            "scores": scores,
            "latency_ms": latency_ms,
        })

        successful += 1

        # ------------------------------------------------------------
        # Progress
        # ------------------------------------------------------------

        if successful % 100 == 0:

            print(
                f"Processed "
                f"{successful}/{len(image_files)}"
            )

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    if successful == 0:
        raise RuntimeError(
            "No images were successfully processed."
        )

    total_time = total_fpga_time

    mean_latency = (
        total_time / successful
    ) * 1000.0

    throughput = (
        successful / total_time
    )

    # ------------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------------

    payload = {
        "model_path": model_path,

        "input_shape": input_shape,
        "input_fix_point": input_fix_point,
        "input_scale": input_scale,

        "output_shape": output_shape,
        "output_fix_point": output_fix_point,
        "output_scale": output_scale,

        "num_samples": successful,

        "total_fpga_time_s": total_time,

        "mean_latency_ms": mean_latency,

        "fpga_throughput_fps": throughput,

        "results": results,
    }

    with open(output_pkl, "wb") as f:
        pickle.dump(
            payload,
            f
        )

    # ------------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FPGA INFERENCE COMPLETE")
    print("=" * 80)

    print(
        f"Samples:             {successful}"
    )

    print(
        f"Total FPGA time:     {total_time:.4f} s"
    )

    print(
        f"Mean FPGA latency:   {mean_latency:.4f} ms"
    )

    print(
        f"FPGA throughput:     {throughput:.2f} FPS"
    )

    print(
        f"Predictions saved:   {output_pkl}"
    )

    print("=" * 80)


# ============================================================================
# Arguments
# ============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "MNIST FPGA/DPU inference. "
            "Images are stored in one directory and "
            "ground-truth labels are provided by CSV."
        )
    )

    parser.add_argument(
        "model_path",
        help="Path to deployed .xmodel"
    )

    parser.add_argument(
        "image_dir",
        help="Directory containing all test images"
    )

    parser.add_argument(
        "csv_path",
        help="CSV containing filename,label"
    )

    parser.add_argument(
        "output_pkl",
        help="Output pickle containing FPGA predictions"
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":

    args = parse_args()

    run_fpga_test(
        model_path=args.model_path,
        image_dir=args.image_dir,
        csv_path=args.csv_path,
        output_pkl=args.output_pkl,
    )
