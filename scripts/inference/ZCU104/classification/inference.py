#!/usr/bin/env python3

from __future__ import print_function

import argparse
import os
import time

import cv2
import numpy as np
import vart
import xir


# =============================================================================
# DPU helper
# =============================================================================

def get_child_subgraph_dpu(graph):
    root_subgraph = graph.get_root_subgraph()
    child_subgraphs = root_subgraph.toposort_child_subgraph()

    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device")
        and cs.get_attr("device").upper() == "DPU"
    ]


# =============================================================================
# MNIST preprocessing
# =============================================================================

def preprocess_mnist(
    image,
    target_width,
    target_height,
    input_scale,
):
    """
    Must match training:

        transforms.ToTensor()
        transforms.Normalize(
            (0.1307,),
            (0.3081,)
        )

    Pipeline:

        uint8 [0,255]
            ->
        float [0,1]
            ->
        normalize
            ->
        quantize
            ->
        int8
    """

    # Convert to grayscale if necessary
    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    # Resize if necessary
    if (
        image.shape[0] != target_height
        or image.shape[1] != target_width
    ):
        image = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )

    # -------------------------------------------------------------------------
    # ToTensor()
    # uint8 [0,255] -> float [0,1]
    # -------------------------------------------------------------------------

    image = image.astype(
        np.float32
    ) / 255.0

    # -------------------------------------------------------------------------
    # Training normalization
    # -------------------------------------------------------------------------

    image = (
        image - 0.1307
    ) / 0.3081

    # -------------------------------------------------------------------------
    # Quantization
    # -------------------------------------------------------------------------

    image = image * input_scale

    # Saturate to int8 range
    image = np.clip(
        image,
        -128,
        127
    )

    return image.astype(np.int8)


# =============================================================================
# Main inference
# =============================================================================

def run_inference(
    model_path,
    image_path,
    output_dir,
    top_k,
):

    # =========================================================================
    # Load XModel
    # =========================================================================

    print(
        f"Loading XModel: {model_path}"
    )

    graph = xir.Graph.deserialize(
        model_path
    )

    # =========================================================================
    # Find DPU
    # =========================================================================

    subgraphs = get_child_subgraph_dpu(
        graph
    )

    if not subgraphs:
        raise RuntimeError(
            "No DPU subgraph found in XModel."
        )

    print(
        f"DPU subgraph: "
        f"{subgraphs[0].get_name()}"
    )

    # =========================================================================
    # Create VART runner
    # =========================================================================

    runner = vart.Runner.create_runner(
        subgraphs[0],
        "run"
    )

    # =========================================================================
    # Get tensors
    # =========================================================================

    input_tensors = (
        runner.get_input_tensors()
    )

    output_tensors = (
        runner.get_output_tensors()
    )

    if len(input_tensors) != 1:
        raise RuntimeError(
            f"Expected 1 input tensor, "
            f"got {len(input_tensors)}"
        )

    if len(output_tensors) != 1:
        raise RuntimeError(
            f"Expected 1 output tensor, "
            f"got {len(output_tensors)}"
        )

    input_tensor = input_tensors[0]
    output_tensor = output_tensors[0]

    # =========================================================================
    # Tensor information
    # =========================================================================

    input_shape = tuple(
        input_tensor.dims
    )

    output_shape = tuple(
        output_tensor.dims
    )

    input_fix_point = (
        input_tensor.get_attr(
            "fix_point"
        )
    )

    output_fix_point = (
        output_tensor.get_attr(
            "fix_point"
        )
    )

    input_scale = float(
        2 ** input_fix_point
    )

    output_scale = float(
        2 ** output_fix_point
    )

    print()
    print("=" * 70)

    print("INPUT")
    print(
        f"  shape:     {input_shape}"
    )
    print(
        f"  fix_point: {input_fix_point}"
    )
    print(
        f"  scale:     {input_scale}"
    )

    print()

    print("OUTPUT")
    print(
        f"  shape:     {output_shape}"
    )
    print(
        f"  fix_point: {output_fix_point}"
    )
    print(
        f"  scale:     {output_scale}"
    )

    print("=" * 70)

    # =========================================================================
    # Validate expected MNIST model
    # =========================================================================

    if input_shape != (1, 28, 28, 1):
        raise RuntimeError(
            "This inference script expects an MNIST "
            f"input shape of (1, 28, 28, 1), "
            f"but the XModel has {input_shape}"
        )

    if output_shape != (1, 10):
        raise RuntimeError(
            "This inference script expects 10 output "
            f"classes, but the XModel has {output_shape}"
        )

    # =========================================================================
    # Read image
    # =========================================================================

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    print()
    print(
        f"Image: {image_path}"
    )

    print(
        f"Original image shape: "
        f"{image.shape}"
    )

    # Keep original image for annotation
    original_image = image.copy()

    # =========================================================================
    # Preprocess
    # =========================================================================

    processed = preprocess_mnist(
        image=image,
        target_width=28,
        target_height=28,
        input_scale=input_scale,
    )

    print()
    print("Preprocessed input:")
    print(
        f"  shape: {processed.shape}"
    )
    print(
        f"  min:   {processed.min()}"
    )
    print(
        f"  max:   {processed.max()}"
    )

    # =========================================================================
    # Prepare DPU input
    # =========================================================================

    input_data = np.empty(
        input_shape,
        dtype=np.int8,
        order="C"
    )

    input_data[
        0, :, :, 0
    ] = processed

    # =========================================================================
    # Prepare DPU output
    # =========================================================================

    output_data = np.empty(
        output_shape,
        dtype=np.int8,
        order="C"
    )

    # =========================================================================
    # Run FPGA inference
    # =========================================================================

    start_total = time.perf_counter()

    start_fpga = time.perf_counter()

    job_id = runner.execute_async(
        [input_data],
        [output_data]
    )

    runner.wait(job_id)

    end_fpga = time.perf_counter()

    end_total = time.perf_counter()

    # =========================================================================
    # Raw output
    # =========================================================================

    raw_output = (
        output_data[0]
        .reshape(-1)
    )

    # =========================================================================
    # Dequantize output
    # =========================================================================

    scores = (
        raw_output.astype(
            np.float32
        ) / output_scale
    )

    # =========================================================================
    # Prediction
    # =========================================================================

    predicted_class = int(
        np.argmax(scores)
    )

    # =========================================================================
    # Top-K
    # =========================================================================

    k = min(
        top_k,
        len(scores)
    )

    top_indices = np.argsort(
        scores
    )[::-1][:k]

    # =========================================================================
    # Timing
    # =========================================================================

    fpga_latency = (
        end_fpga - start_fpga
    ) * 1000.0

    end_to_end_latency = (
        end_total - start_total
    ) * 1000.0

    # =========================================================================
    # Print results
    # =========================================================================

    print()
    print("=" * 70)

    print("Raw FPGA output:")
    print(raw_output)

    print()

    print("Dequantized output:")
    print(scores)

    print()

    print(
        "======================================================================"
    )

    print(
        f"Predicted digit: {predicted_class}"
    )

    print(
        f"Model output score: "
        f"{scores[predicted_class]:.4f}"
    )

    print("-" * 70)

    print(
        f"FPGA latency:       "
        f"{fpga_latency:.2f} ms"
    )

    if fpga_latency > 0:
        print(
            f"FPGA FPS:            "
            f"{1000.0 / fpga_latency:.2f}"
        )

    print(
        f"End-to-end latency: "
        f"{end_to_end_latency:.2f} ms"
    )

    if end_to_end_latency > 0:
        print(
            f"End-to-end FPS:      "
            f"{1000.0 / end_to_end_latency:.2f}"
        )

    print(
        f"Non-FPGA overhead:  "
        f"{end_to_end_latency - fpga_latency:.2f} ms"
    )

    print("-" * 70)

    print(
        f"Top-{k} predictions:"
    )

    for rank, class_idx in enumerate(
        top_indices,
        start=1
    ):

        print(
            f"  {rank}. "
            f"{int(class_idx)}: "
            f"{scores[class_idx]:.4f}"
        )

    print(
        "======================================================================"
    )

    # =========================================================================
    # Save annotated image
    # =========================================================================

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    annotated = cv2.cvtColor(
        original_image,
        cv2.COLOR_GRAY2BGR
    )

    cv2.putText(
        annotated,
        f"Predicted: {predicted_class}",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    for rank, class_idx in enumerate(
        top_indices,
        start=1
    ):

        text = (
            f"{rank}. {int(class_idx)} "
            f"{scores[class_idx]:.2f}"
        )

        cv2.putText(
            annotated,
            text,
            (5, 20 + rank * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_path = os.path.join(
        output_dir,
        f"{base_name}_annotated.jpg"
    )

    cv2.imwrite(
        output_path,
        annotated
    )

    print()
    print(
        f"Annotated image saved to: "
        f"{output_path}"
    )


# =============================================================================
# Arguments
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Single-image MNIST classification "
            "inference on FPGA using VART"
        )
    )

    parser.add_argument(
        "model_path",
        help="Path to the .xmodel"
    )

    parser.add_argument(
        "image_path",
        help="Path to a single MNIST image"
    )

    parser.add_argument(
        "output_dir",
        nargs="?",
        default="results",
        help=(
            "Directory for annotated output "
            "(default: results)"
        )
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help=(
            "Number of top predictions "
            "(default: 5)"
        )
    )

    return parser.parse_args()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    args = parse_args()

    run_inference(
        model_path=args.model_path,
        image_path=args.image_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
