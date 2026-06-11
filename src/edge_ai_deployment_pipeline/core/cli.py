from __future__ import annotations

import argparse
import logging
import sys

from pathlib import Path

from edge_ai_deployment_pipeline.core.logging_config import setup_logging
from edge_ai_deployment_pipeline.core.pipeline import Pipeline
from edge_ai_deployment_pipeline.core.pytorch_loader import parse_model_kwargs
from edge_ai_deployment_pipeline.core.types import LatencyType, UserRequirements

logger = logging.getLogger(__name__)


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
        description="Profile a CV model, rank edge boards, and emit a deployable artifact.",
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Weights (.pt/.pth state_dict) or Keras model (.keras, .h5)",
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
        help="Path to platforms.json (default: bundled config/catalogue platforms.json)",
    )
    parser.add_argument(
        "--latency-type",
        type=LatencyType,
        required=True,
        choices=list(LatencyType),
        help="Target latency class: ultra_low (FPGA), low (MCU), medium (Jetson), moderate (ARM)",
    )
    parser.add_argument(
        "--max-power-w",
        type=float,
        default=25.0,
        help="Max board power budget in W (default: 25)",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=500.0,
        help="Max board cost budget in USD (default: 500)",
    )
    parser.add_argument(
        "--toolchain",
        type=str,
        default=None,
        help="Force toolchain (stm32n6, vitis_ai, tensorrt, ncnn, …)",
    )
    parser.add_argument(
        "--board-id",
        type=str,
        default=None,
        help="Deploy to this board id (skips interactive board selection)",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Deploy to the highest-ranked board without prompting",
    )
    parser.add_argument(
        "--model-class",
        type=str,
        default=None,
        required=False,
        metavar="MODULE:CLASS",
        help=(
            "Required for .pt/.pth weights: network definition "
            "(e.g. data/models/cifar_net.py:Net)"
        ),
    )
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default=None,
        metavar="JSON",
        help='Optional JSON kwargs for the model constructor (e.g. \'{"num_classes": 10}\')',
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
        logger.error("Model file not found: %s", args.model)
        return 1

    if args.model.suffix.lower() in {".pt", ".pth"} and not args.model_class:
        logger.error(
            "PyTorch weights (%s) require --model-class (e.g. data/models/cifar_net.py:Net)",
            args.model,
        )
        return 1

    requirements = UserRequirements(
        latency_type=args.latency_type,
        max_power_watts=args.max_power_w,
        max_cost_usd=args.max_cost_usd,
        toolchain=args.toolchain,
    )

    try:
        result = Pipeline(
            model_path=args.model,
            input_shape=args.input_shape,
            requirements=requirements,
            output_dir=args.output_dir,
            platform_config_path=args.platform_config,
            board_id=args.board_id,
            auto_select_board=args.auto_select,
            model_class=args.model_class,
            model_kwargs=parse_model_kwargs(args.model_kwargs),
        ).run()
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Board selection cancelled")
        return 1

    sys.stdout.write(result.to_json())
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
