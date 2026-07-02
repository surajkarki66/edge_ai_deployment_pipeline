from __future__ import annotations

import argparse
import logging
import sys

from pathlib import Path

from edge_ai_deployment_pipeline.config.paths import DEFAULT_ARTIFACTS_DIR, EXAMPLE_MODEL_CLASS
from edge_ai_deployment_pipeline.core.host_check import doctor_main
from edge_ai_deployment_pipeline.core.logging_config import setup_logging
from edge_ai_deployment_pipeline.core.pipeline import Pipeline
from edge_ai_deployment_pipeline.core.pytorch_loader import parse_model_kwargs
from edge_ai_deployment_pipeline.core.run_context import (
    clean_workspace,
    new_run_id,
    rotate_artifact_runs,
    rotate_logs,
    run_artifacts_dir,
)
from edge_ai_deployment_pipeline.core.types import CvTask, LatencyType, UserRequirements

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


def _parse_deploy_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a CV model, rank edge boards, and emit a deployable artifact.",
    )
    parser.add_argument(
        "model",
        type=Path,
        nargs="?",
        help="Weights (.pt/.pth state_dict) or Keras model (.keras, .h5)",
    )
    parser.add_argument(
        "--input-shape",
        type=_parse_input_shape,
        required=False,
        metavar="C,H,W",
        help="Model input as channels,height,width (e.g. 3,224,224)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Base directory for run artifacts (default: data/artifacts/<run-id>/)",
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
        required=False,
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
        "--task",
        type=CvTask,
        required=False,
        choices=list(CvTask),
        help="CV task: image_classification or object_detection",
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
            f"(e.g. {EXAMPLE_MODEL_CLASS})"
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
        "--check-host",
        action="store_true",
        help="Run host pre-flight checks (Docker, Vitis-AI, STEdgeAI, TensorRT, CUDA) and exit",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous artifact runs and logs before starting",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def _run_deploy(argv: list[str]) -> int:
    args = _parse_deploy_args(argv)

    if args.check_host:
        return doctor_main([])

    if args.model is None:
        parser.error("model path is required (unless using --check-host)")

    if args.input_shape is None:
        parser.error("--input-shape is required")
    if args.latency_type is None:
        parser.error("--latency-type is required")
    if args.task is None:
        parser.error("--task is required")

    run_id = new_run_id()
    artifacts_base = args.output_dir
    if args.clean:
        clean_workspace(artifacts_base=artifacts_base)
    else:
        rotate_logs()
        rotate_artifact_runs(artifacts_base)

    run_output_dir = run_artifacts_dir(artifacts_base, run_id)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(verbose=args.verbose, log_file=f"pipeline_{run_id}.log")
    logger.info("Run id: %s", run_id)
    logger.info("Artifacts: %s", run_output_dir)

    if not args.model.is_file():
        logger.error("Model file not found: %s", args.model)
        return 1

    if (
        args.model.suffix.lower() in {".pt", ".pth"}
        and not args.model_class
        and args.task != CvTask.OBJECT_DETECTION
    ):
        logger.error(
            f"PyTorch weights (%s) require --model-class (e.g. {EXAMPLE_MODEL_CLASS}), "
            "or pass --task object_detection for an Ultralytics checkpoint",
            args.model,
        )
        return 1

    requirements = UserRequirements(
        latency_type=args.latency_type,
        task=args.task,
        max_power_watts=args.max_power_w,
        max_cost_usd=args.max_cost_usd,
        toolchain=args.toolchain,
    )

    try:
        result = Pipeline(
            model_path=args.model,
            input_shape=args.input_shape,
            requirements=requirements,
            output_dir=run_output_dir,
            platform_config_path=args.platform_config,
            board_id=args.board_id,
            auto_select_board=args.auto_select,
            model_class=args.model_class,
            model_kwargs=parse_model_kwargs(args.model_kwargs),
            run_id=run_id,
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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "doctor":
        return doctor_main(argv[1:])
    return _run_deploy(argv)


if __name__ == "__main__":
    raise SystemExit(main())
