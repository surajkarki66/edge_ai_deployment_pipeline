from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STM32_AI_EXE = Path(
    "/opt/ST/STEdgeAI/4.0/Utilities/linux/stedgeai"
)
DEFAULT_STM32_CUBE_IDE_EXE = Path(
    "/opt/st/stm32cubeide_2.1.1/stm32cubeide"
)


@dataclass(frozen=True)
class HostCheckResult:
    check_id: str
    display_name: str
    ok: bool
    message: str
    hint: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        payload: dict[str, str | bool] = {
            "check_id": self.check_id,
            "display_name": self.display_name,
            "ok": self.ok,
            "message": self.message,
        }
        if self.hint:
            payload["hint"] = self.hint
        return payload


def _ok(check_id: str, name: str, message: str) -> HostCheckResult:
    return HostCheckResult(check_id, name, True, message)


def _fail(check_id: str, name: str, message: str, hint: str = "") -> HostCheckResult:
    return HostCheckResult(check_id, name, False, message, hint)


def check_python_packages() -> HostCheckResult:
    name = "Python stack (torch, tensorflow)"
    try:
        import torch
        import tensorflow  # noqa: F401
    except ImportError as exc:
        return _fail(
            "python_packages",
            name,
            f"Missing dependency: {exc}",
            "Run: pip install -e .",
        )
    return _ok("python_packages", name, f"torch {torch.__version__}, tensorflow available")


def check_docker() -> HostCheckResult:
    name = "Docker (FPGA / Vitis AI)"
    if shutil.which("docker") is None:
        return _fail(
            "docker",
            name,
            "docker not found on PATH",
            "Install Docker and add your user to the docker group.",
        )
    return _ok("docker", name, f"docker at {shutil.which('docker')}")


VITIS_AI_IMAGE = "xilinx/vitis-ai-pytorch-cpu:latest"


def check_vitis_ai_docker_image() -> HostCheckResult:
    name = "Vitis-AI Docker image (FPGA)"
    if shutil.which("docker") is None:
        return _fail(
            "vitis_ai_docker",
            name,
            "docker not available",
            "Install Docker before pulling the Vitis-AI image.",
        )
    result = subprocess.run(
        ["docker", "image", "inspect", VITIS_AI_IMAGE],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return _ok("vitis_ai_docker", name, VITIS_AI_IMAGE)
    return _fail(
        "vitis_ai_docker",
        name,
        f"image not present locally: {VITIS_AI_IMAGE}",
        f"Run: docker pull {VITIS_AI_IMAGE}",
    )


def check_stm32_ai_exe() -> HostCheckResult:
    name = "STEdgeAI (MCU)"
    raw = os.environ.get("STM32_AI_EXE", str(DEFAULT_STM32_AI_EXE))
    path = Path(raw)
    if path.is_file() and os.access(path, os.X_OK):
        return _ok("stm32_ai_exe", name, str(path.resolve()))
    return _fail(
        "stm32_ai_exe",
        name,
        f"stedgeai not executable: {path}",
        "Install STEdgeAI and export STM32_AI_EXE=/path/to/stedgeai",
    )


def check_stm32_cube_ide() -> HostCheckResult:
    name = "STM32CubeIDE (MCU firmware build)"
    raw = os.environ.get("STM32_CUBE_IDE_EXE", str(DEFAULT_STM32_CUBE_IDE_EXE))
    path = Path(raw)
    if path.is_file() and os.access(path, os.X_OK):
        return _ok("stm32_cube_ide", name, str(path.resolve()))
    return _fail(
        "stm32_cube_ide",
        name,
        f"STM32CubeIDE not executable: {path}",
        "Install STM32CubeIDE and export STM32_CUBE_IDE_EXE=/path/to/stm32cubeide",
    )


def check_tensorrt() -> HostCheckResult:
    name = "TensorRT 10.x (Jetson INT8)"
    try:
        import tensorrt as trt
    except ImportError:
        return _fail(
            "tensorrt",
            name,
            "tensorrt package not installed",
            "pip install 'tensorrt>=10.16.1.11,<11'",
        )
    major = int(str(trt.__version__).split(".", 1)[0])
    if major >= 11:
        return _fail(
            "tensorrt",
            name,
            f"TensorRT {trt.__version__} found; INT8 calibration needs 10.x",
            "pip install 'tensorrt>=10.16.1.11,<11'",
        )
    return _ok("tensorrt", name, f"TensorRT {trt.__version__}")


def check_cuda() -> HostCheckResult:
    name = "CUDA (Jetson TensorRT build)"
    try:
        import torch
    except ImportError:
        return _fail("cuda", name, "torch not installed", "pip install torch")
    if torch.cuda.is_available():
        return _ok(
            "cuda",
            name,
            f"CUDA available ({torch.cuda.get_device_name(0)})",
        )
    return _fail(
        "cuda",
        name,
        "No CUDA GPU detected for TensorRT engine build",
        "Build on an NVIDIA host or on the target Jetson device.",
    )


_ALL_CHECKS: dict[str, object] = {
    "python_packages": check_python_packages,
    "docker": check_docker,
    "vitis_ai_docker": check_vitis_ai_docker_image,
    "stm32_ai_exe": check_stm32_ai_exe,
    "stm32_cube_ide": check_stm32_cube_ide,
    "tensorrt": check_tensorrt,
    "cuda": check_cuda,
}

_DEFAULT_CHECK_IDS = (
    "python_packages",
    "docker",
    "vitis_ai_docker",
    "stm32_ai_exe",
    "stm32_cube_ide",
    "tensorrt",
    "cuda",
)


def run_host_checks(check_ids: list[str] | None = None) -> list[HostCheckResult]:
    ids = check_ids or list(_DEFAULT_CHECK_IDS)
    results: list[HostCheckResult] = []
    for check_id in ids:
        fn = _ALL_CHECKS.get(check_id)
        if fn is None:
            results.append(
                _fail(check_id, check_id, f"Unknown check '{check_id}'"),
            )
            continue
        results.append(fn())  # type: ignore[operator]
    return results


def format_host_check_report(results: list[HostCheckResult]) -> str:
    lines = ["Host pre-flight checks:"]
    for result in results:
        status = "OK" if result.ok else "FAIL"
        lines.append(f"  [{status}] {result.display_name}: {result.message}")
        if not result.ok and result.hint:
            lines.append(f"         hint: {result.hint}")
    failed = sum(1 for r in results if not r.ok)
    lines.append("")
    lines.append(f"Summary: {len(results) - failed}/{len(results)} passed")
    return "\n".join(lines)


def doctor_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify host tools for edge-ai-deploy (Docker, Vitis-AI, STEdgeAI, STM32CubeIDE, TensorRT, CUDA).",
    )
    parser.add_argument(
        "--check",
        action="append",
        dest="checks",
        metavar="ID",
        help=f"Run specific check(s): {', '.join(_DEFAULT_CHECK_IDS)}",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print failures; exit 1 if any check fails",
    )
    args = parser.parse_args(argv)

    results = run_host_checks(args.checks)
    if args.quiet:
        failures = [r for r in results if not r.ok]
        for result in failures:
            sys.stderr.write(f"{result.display_name}: {result.message}\n")
            if result.hint:
                sys.stderr.write(f"  hint: {result.hint}\n")
        return 1 if failures else 0

    report = format_host_check_report(results)
    sys.stdout.write(report)
    sys.stdout.write("\n")
    return 0 if all(r.ok for r in results) else 1


def main(argv: list[str] | None = None) -> int:
    return doctor_main(argv)
