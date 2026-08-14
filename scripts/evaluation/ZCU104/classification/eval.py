#!/usr/bin/env python3
import argparse
import json
import pickle
from collections import defaultdict

import numpy as np


def run_evaluate(predictions_pkl, report_path=None):
    with open(predictions_pkl, "rb") as f:
        payload = pickle.load(f)

    results = payload["results"]
    class_names = payload.get("class_names")
    n = len(results)
    if n == 0:
        raise RuntimeError("No predictions found in the pickle.")

    correct = 0
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))  # confusion[true][pred] = count
    latencies = []

    for r in results:
        t, p = r["true_label"], r["pred_label"]
        latencies.append(r["latency_ms"])
        per_class_total[t] += 1
        confusion[t][p] += 1
        if t == p:
            correct += 1
            per_class_correct[t] += 1

    accuracy = correct / n
    latencies = np.array(latencies)

    print("=" * 80)
    print(f"Samples evaluated: {n}")
    print(f"Overall accuracy:  {accuracy*100:.2f}%  ({correct}/{n})")
    print(f"Latency (ms): mean={latencies.mean():.2f} "
          f"p50={np.percentile(latencies,50):.2f} "
          f"p95={np.percentile(latencies,95):.2f} "
          f"max={latencies.max():.2f}")
    print(f"Throughput: {1000.0/latencies.mean():.2f} FPS (per-sample DPU latency)")
    print("-" * 80)
    print("Per-class accuracy:")

    per_class_report = {}
    for cls in sorted(per_class_total.keys(), key=lambda x: str(x)):
        tot = per_class_total[cls]
        cor = per_class_correct.get(cls, 0)
        acc = cor / tot if tot else 0.0
        name = class_names[cls] if class_names and isinstance(cls, int) and cls < len(class_names) else str(cls)
        print(f"  class {name}: {acc*100:6.2f}%  ({cor}/{tot})")
        per_class_report[str(cls)] = {"accuracy": acc, "correct": cor, "total": tot}

    print("-" * 80)
    print("Confusion matrix (rows=true, cols=pred):")
    labels_sorted = sorted(per_class_total.keys(), key=lambda x: str(x))
    header = "      " + " ".join(f"{str(l):>4}" for l in labels_sorted)
    print(header)
    for t in labels_sorted:
        row = " ".join(f"{confusion[t].get(p,0):4d}" for p in labels_sorted)
        print(f"{str(t):>5} {row}")

    if report_path:
        report = {
            "num_samples": n,
            "accuracy": accuracy,
            "correct": correct,
            "latency_ms": {
                "mean": float(latencies.mean()),
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
                "max": float(latencies.max()),
            },
            "per_class": per_class_report,
            "confusion_matrix": {str(t): {str(p): c for p, c in row.items()} for t, row in confusion.items()},
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report saved to: {report_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute accuracy/metrics from FPGA predictions saved by mnist_fpga_predict.py"
    )
    p.add_argument("predictions_pkl", help="Path to predictions pickle from the predict script")
    p.add_argument("--report", default=None, help="Optional path to write a JSON report")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluate(args.predictions_pkl, args.report)