"""Run the edge deployment pipeline on a trained computer-vision model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from core.pipeline import Pipeline
from core.types import UserRequirements


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a CV model, pick an edge board, and emit a deployable artifact.",
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Path to trained model (.pt, .pth, .onnx, .keras, .h5, …)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for deployment artifacts (default: artifacts)",
    )
    parser.add_argument(
        "--platform-config",
        type=Path,
        default=None,
        help="Path to platforms.json (default: config/platforms.json)",
    )
    parser.add_argument("--max-latency-ms", type=float, default=None, help="Max inference latency (ms)")
    parser.add_argument("--max-power-w", type=float, default=None, help="Max board power (W)")
    parser.add_argument("--max-cost-usd", type=float, default=None, help="Max board cost (USD)")
    parser.add_argument(
        "--toolchain",
        type=str,
        default=None,
        help="Force toolchain (tensorrt, vitis_ai, stm32cubeai, ncnn)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.model.is_file():
        print(f"Model file not found: {args.model}", file=sys.stderr)
        return 1

    requirements = UserRequirements(
        max_latency_ms=args.max_latency_ms,
        max_power_watts=args.max_power_w,
        max_cost_usd=args.max_cost_usd,
        toolchain=args.toolchain,
    )

    result = Pipeline(
        model_path=args.model,
        requirements=requirements,
        output_dir=args.output_dir,
        platform_config_path=args.platform_config,
    ).run()

    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
