from ultralytics import YOLO
import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an Ultralytics YOLO model on a dataset split."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the YOLO model."
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (default: test)."
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)."
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size (default: 1)."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device, e.g. cpu, 0, 0,1 (default: cpu)."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="test_results.json",
        help="Output JSON file (default: test_results.json)."
    )

    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)

    # Evaluate
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    # Extract metrics
    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "mAP75": float(results.box.map75),
    }

    # Save results
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=4))
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
