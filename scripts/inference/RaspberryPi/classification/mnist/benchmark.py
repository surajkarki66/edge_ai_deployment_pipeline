import argparse
import os
import time
import gc

import cv2
import numpy as np
import ncnn


# ============================================================
# Load NCNN model
# ============================================================
def load_model(param_path, bin_path):

    if not os.path.isfile(param_path):
        raise FileNotFoundError(param_path)

    if not os.path.isfile(bin_path):
        raise FileNotFoundError(bin_path)

    net = ncnn.Net()

    # IMPORTANT:
    # FP16 caused NaN/Inf on this Raspberry Pi.
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_arithmetic = False

    ret = net.load_param(param_path)

    if ret != 0:
        raise RuntimeError(
            f"Failed to load param file: {ret}"
        )

    ret = net.load_model(bin_path)

    if ret != 0:
        raise RuntimeError(
            f"Failed to load bin file: {ret}"
        )

    return net


# ============================================================
# Load image
# ============================================================
def load_image(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    return image


# ============================================================
# Preprocess
# ============================================================
def preprocess(image):

    image = cv2.resize(
        image,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = (
        image - 0.1307
    ) / 0.3081

    return image


# ============================================================
# Create NCNN extractor
# ============================================================
def create_extractor(net):

    ex = net.create_extractor()

    ex.set_light_mode(True)

    return ex


# ============================================================
# Detector-only inference
# ============================================================
def detector_only(net, image):

    mat = ncnn.Mat(image)

    ex = create_extractor(net)

    start = time.perf_counter()

    ret = ex.input(
        "in0",
        mat
    )

    if ret != 0:
        raise RuntimeError(
            f"NCNN input failed: {ret}"
        )

    ret, out = ex.extract(
        "out0"
    )

    end = time.perf_counter()

    if ret != 0:
        raise RuntimeError(
            f"NCNN extraction failed: {ret}"
        )

    # Make sure output is consumed
    output = np.array(
        out,
        dtype=np.float32
    ).flatten()

    # Check output
    if not np.all(np.isfinite(output)):
        raise RuntimeError(
            "NCNN produced NaN/Inf during benchmark"
        )

    latency_ms = (
        end - start
    ) * 1000.0

    prediction = int(
        np.argmax(output)
    )

    return latency_ms, prediction


# ============================================================
# End-to-end inference
#
# Image is already in RAM.
#
# Includes:
#   resize
#   normalization
#   NCNN inference
#   output conversion
#   argmax
#
# Does NOT include disk I/O.
# ============================================================
def end_to_end(net, original_image):

    start = time.perf_counter()

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------
    image = cv2.resize(
        original_image,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = (
        image - 0.1307
    ) / 0.3081

    # --------------------------------------------------------
    # NCNN
    # --------------------------------------------------------
    mat = ncnn.Mat(image)

    ex = create_extractor(net)

    ret = ex.input(
        "in0",
        mat
    )

    if ret != 0:
        raise RuntimeError(
            f"NCNN input failed: {ret}"
        )

    ret, out = ex.extract(
        "out0"
    )

    if ret != 0:
        raise RuntimeError(
            f"NCNN extraction failed: {ret}"
        )

    # --------------------------------------------------------
    # Postprocessing
    # --------------------------------------------------------
    output = np.array(
        out,
        dtype=np.float32
    ).flatten()

    if not np.all(np.isfinite(output)):
        raise RuntimeError(
            "NCNN produced NaN/Inf during benchmark"
        )

    prediction = int(
        np.argmax(output)
    )

    end = time.perf_counter()

    latency_ms = (
        end - start
    ) * 1000.0

    return latency_ms, prediction


# ============================================================
# Full file-to-prediction benchmark
#
# Includes:
#   disk read
#   resize
#   normalization
#   NCNN
#   postprocessing
# ============================================================
def full_e2e(net, image_path):

    start = time.perf_counter()

    # --------------------------------------------------------
    # Disk I/O
    # --------------------------------------------------------
    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------
    image = cv2.resize(
        image,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = (
        image - 0.1307
    ) / 0.3081

    # --------------------------------------------------------
    # NCNN
    # --------------------------------------------------------
    mat = ncnn.Mat(image)

    ex = create_extractor(net)

    ret = ex.input(
        "in0",
        mat
    )

    if ret != 0:
        raise RuntimeError(
            f"NCNN input failed: {ret}"
        )

    ret, out = ex.extract(
        "out0"
    )

    if ret != 0:
        raise RuntimeError(
            f"NCNN extraction failed: {ret}"
        )

    # --------------------------------------------------------
    # Postprocessing
    # --------------------------------------------------------
    output = np.array(
        out,
        dtype=np.float32
    ).flatten()

    if not np.all(np.isfinite(output)):
        raise RuntimeError(
            "NCNN produced NaN/Inf"
        )

    prediction = int(
        np.argmax(output)
    )

    end = time.perf_counter()

    latency_ms = (
        end - start
    ) * 1000.0

    return latency_ms, prediction


# ============================================================
# Statistics
# ============================================================
def statistics(times):

    times = np.asarray(
        times,
        dtype=np.float64
    )

    mean = np.mean(times)

    return {
        "mean": mean,
        "median": np.median(times),
        "p95": np.percentile(times, 95),
        "p99": np.percentile(times, 99),
        "min": np.min(times),
        "max": np.max(times),
        "fps": 1000.0 / mean,
    }


# ============================================================
# Print results
# ============================================================
def print_results(title, stats):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(
        f"Mean latency   : {stats['mean']:.4f} ms"
    )

    print(
        f"Median latency : {stats['median']:.4f} ms"
    )

    print(
        f"P95 latency    : {stats['p95']:.4f} ms"
    )

    print(
        f"P99 latency    : {stats['p99']:.4f} ms"
    )

    print(
        f"Min latency    : {stats['min']:.4f} ms"
    )

    print(
        f"Max latency    : {stats['max']:.4f} ms"
    )

    print(
        f"FPS            : {stats['fps']:.2f}"
    )

    print("=" * 60)


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--param",
        required=True
    )

    parser.add_argument(
        "--bin",
        required=True
    )

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    print("\nLoading NCNN model...")

    net = load_model(
        args.param,
        args.bin
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Load image once
    # --------------------------------------------------------
    original_image = load_image(
        args.image
    )

    processed_image = preprocess(
        original_image
    )

    print(
        f"\nBenchmark image: {args.image}"
    )

    print(
        f"Warm-up: {args.warmup}"
    )

    print(
        f"Iterations: {args.iterations}"
    )

    # ========================================================
    # WARM-UP
    # ========================================================
    print("\nRunning warm-up...")

    for _ in range(args.warmup):

        detector_only(
            net,
            processed_image
        )

    print("Warm-up completed.")

    # Force Python garbage collection
    gc.collect()

    # ========================================================
    # DETECTOR ONLY
    # ========================================================
    print(
        "\nRunning detector-only benchmark..."
    )

    detector_times = []

    for i in range(args.iterations):

        latency, prediction = detector_only(
            net,
            processed_image
        )

        detector_times.append(
            latency
        )

        if (i + 1) % 100 == 0:

            print(
                f"Detector: "
                f"{i + 1}/{args.iterations}"
            )

    detector_stats = statistics(
        detector_times
    )

    # ========================================================
    # END-TO-END IN RAM
    # ========================================================
    print(
        "\nRunning end-to-end benchmark "
        "(without disk I/O)..."
    )

    e2e_times = []

    for i in range(args.iterations):

        latency, prediction = end_to_end(
            net,
            original_image
        )

        e2e_times.append(
            latency
        )

        if (i + 1) % 100 == 0:

            print(
                f"E2E: "
                f"{i + 1}/{args.iterations}"
            )

    e2e_stats = statistics(
        e2e_times
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")
    print("#" * 60)
    print("# NCNN BENCHMARK")
    print("#" * 60)

    print_results(
        "DETECTOR-ONLY",
        detector_stats
    )

    print_results(
        "END-TO-END (IMAGE IN RAM)",
        e2e_stats
    )

    # --------------------------------------------------------
    # Overhead
    # --------------------------------------------------------
    overhead = (
        e2e_stats["mean"]
        - detector_stats["mean"]
    )

    print("\n" + "=" * 60)
    print("OVERHEAD")
    print("=" * 60)

    print(
        f"Preprocessing + postprocessing: "
        f"{overhead:.4f} ms"
    )

    print("=" * 60)

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
