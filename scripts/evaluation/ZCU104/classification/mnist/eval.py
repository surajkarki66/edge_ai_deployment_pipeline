#!/usr/bin/env python3

import argparse
import json
import pickle
from collections import defaultdict

import numpy as np


def evaluate(predictions_pkl, report_path=None):

    with open(predictions_pkl, "rb") as f:
        payload = pickle.load(f)

    results = payload["results"]

    if not results:
        raise RuntimeError(
            "No predictions found."
        )

    total = len(results)

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------

    correct = 0

    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)

    confusion = defaultdict(
        lambda: defaultdict(int)
    )

    latencies = []

    for result in results:

        true_label = int(
            result["true_label"]
        )

        pred_label = int(
            result["pred_label"]
        )

        latency = float(
            result["latency_ms"]
        )

        latencies.append(latency)

        per_class_total[true_label] += 1

        confusion[true_label][pred_label] += 1

        if true_label == pred_label:

            correct += 1

            per_class_correct[
                true_label
            ] += 1

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    accuracy = correct / total

    latencies = np.asarray(
        latencies,
        dtype=np.float64
    )

    mean_latency = latencies.mean()

    p50_latency = np.percentile(
        latencies,
        50
    )

    p95_latency = np.percentile(
        latencies,
        95
    )

    max_latency = latencies.max()

    throughput = (
        1000.0 / mean_latency
    )

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("MNIST FPGA EVALUATION")
    print("=" * 80)

    print(
        f"Samples evaluated: {total}"
    )

    print(
        f"Correct:           {correct}"
    )

    print(
        f"Incorrect:         {total - correct}"
    )

    print(
        f"Overall accuracy:  "
        f"{accuracy * 100:.2f}% "
        f"({correct}/{total})"
    )

    print("-" * 80)

    print("FPGA latency:")

    print(
        f"  Mean: {mean_latency:.4f} ms"
    )

    print(
        f"  P50:  {p50_latency:.4f} ms"
    )

    print(
        f"  P95:  {p95_latency:.4f} ms"
    )

    print(
        f"  Max:  {max_latency:.4f} ms"
    )

    print(
        f"  FPGA throughput: "
        f"{throughput:.2f} FPS"
    )

    print("-" * 80)

    # ------------------------------------------------------------------
    # Per-class accuracy
    # ------------------------------------------------------------------

    print("Per-class accuracy:")

    per_class_report = {}

    for label in sorted(
        per_class_total.keys()
    ):

        total_class = (
            per_class_total[label]
        )

        correct_class = (
            per_class_correct.get(
                label,
                0
            )
        )

        class_accuracy = (
            correct_class /
            total_class
        )

        print(
            f"  {label}: "
            f"{correct_class}/{total_class} "
            f"({class_accuracy * 100:.2f}%)"
        )

        per_class_report[str(label)] = {
            "correct": correct_class,
            "total": total_class,
            "accuracy": class_accuracy,
        }

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    labels = sorted(
        per_class_total.keys()
    )

    print()
    print("-" * 80)
    print("Confusion matrix")
    print("Rows = true, columns = predicted")
    print()

    print(
        "     " +
        " ".join(
            f"{label:4d}"
            for label in labels
        )
    )

    for true_label in labels:

        row = []

        for pred_label in labels:

            value = confusion[
                true_label
            ].get(
                pred_label,
                0
            )

            row.append(
                f"{value:4d}"
            )

        print(
            f"{true_label:3d}  " +
            " ".join(row)
        )

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------

    if report_path:

        report = {

            "num_samples": total,

            "correct": correct,

            "incorrect": total - correct,

            "accuracy": accuracy,

            "latency_ms": {
                "mean": float(mean_latency),
                "p50": float(p50_latency),
                "p95": float(p95_latency),
                "max": float(max_latency),
            },

            "throughput_fps": float(
                throughput
            ),

            "per_class": per_class_report,

            "confusion_matrix": {
                str(true_label): {
                    str(pred_label): count
                    for pred_label, count
                    in confusion[true_label].items()
                }
                for true_label in labels
            },
        }

        with open(
            report_path,
            "w"
        ) as f:

            json.dump(
                report,
                f,
                indent=2
            )

        print()
        print(
            f"JSON report saved to: "
            f"{report_path}"
        )

    print("=" * 80)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MNIST FPGA predictions "
            "without using the FPGA."
        )
    )

    parser.add_argument(
        "predictions_pkl",
        help="Prediction pickle from mnist_fpga_test.py"
    )

    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    evaluate(
        predictions_pkl=args.predictions_pkl,
        report_path=args.report,
    )
