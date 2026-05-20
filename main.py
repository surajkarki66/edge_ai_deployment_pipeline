from __future__ import annotations

import argparse
import sys

from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.logging_config import setup_logging
from core.pipeline import Pipeline
from core.types import UserRequirements


def _parse_input_shape(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("input shape must be C,H,W (e.g. 3,224,224)")
    try:
        channels, height, width = (int(p.strip()) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("input shape dimensions must be integers") from exc
    if channels < 1 or height < 1 or width < 1:
        raise argparse.ArgumentTypeError("input shape dimensions must be positive")
    return channels, height, width


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a CV model, pick an edge board, and emit a deployable artifact.",
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Path to trained model (.pt, .pth, .keras, .h5, …)",
    )
    parser.add_argument(
        "--input-shape",
        type=_parse_input_shape,
        required=True,
        metavar="C,H,W",
        help="Model input as channels,height,width (e.g. 3,224,224)",
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
        help="Path to platforms.json (default: src/config/platforms.json)",
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

    setup_logging(verbose=args.verbose)

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
        input_shape=args.input_shape,
        requirements=requirements,
        output_dir=args.output_dir,
        platform_config_path=args.platform_config,
    ).run()

    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
