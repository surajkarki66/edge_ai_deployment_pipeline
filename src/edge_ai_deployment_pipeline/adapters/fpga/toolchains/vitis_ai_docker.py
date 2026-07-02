"""Run Vitis AI PyTorch quantization and DPU compilation inside official Docker images.

Requires a local clone of https://github.com/Xilinx/Vitis-AI and Docker.
See: https://github.com/Xilinx/Vitis-AI
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import time


from pathlib import Path
from typing import Any

from edge_ai_deployment_pipeline.adapters.fpga.toolchains.ultralytics_dpu.patch_ultralytics import (
    VITIS_ULTRALYTICS_VERSION,
)

logger = logging.getLogger(__name__)

PYTORCH_IMAGE = "xilinx/vitis-ai-pytorch-cpu:latest"

CONDA_ACTIVATE_PYTORCH = (
    "source /opt/vitis_ai/conda/bin/activate vitis-ai-pytorch"
)

_TOOLCHAIN_DIR = Path(__file__).resolve().parent
_ULTRALYTICS_DPU_DIR = _TOOLCHAIN_DIR / "ultralytics_dpu"
_PYTORCH_SCRIPT = _TOOLCHAIN_DIR / "scripts" / "vai_q_pytorch.py"
_PATCH_ULTRALYTICS_SCRIPT = _ULTRALYTICS_DPU_DIR / "patch_ultralytics.py"
_CUSTOM_LAYERS_DIR = _ULTRALYTICS_DPU_DIR / "custom_layers"
_ARCH_DIR = _TOOLCHAIN_DIR / "architectures"

ARCH_FILES = {
    "DPUCZDX8G_ISA1_B512": "arch_B512.json",
    "DPUCZDX8G_ISA1_B800": "arch_B800.json",
    "DPUCZDX8G_ISA1_B1024": "arch_B1024.json",
    "DPUCZDX8G_ISA1_B1152": "arch_B1152.json",
    "DPUCZDX8G_ISA1_B1600": "arch_B1600.json",
    "DPUCZDX8G_ISA1_B2304": "arch_B2304.json",
    "DPUCZDX8G_ISA1_B3136": "arch_B3136.json",
    "DPUCZDX8G_ISA1_B4096": "arch_B4096.json",
}


def resolve_vitis_ai_repo(config: dict[str, Any]) -> Path:
    env = os.environ.get("VITIS_AI_REPO")
    if env:
        return Path(env).resolve()
    if config.get("vitis_ai_repo"):
        return Path(str(config["vitis_ai_repo"])).resolve()
    return Path.cwd() / "Vitis-AI"


def resolve_dpu_target(config: dict[str, Any]) -> str:
    target = config.get("dpu_target") or config.get("dpu_arch_target")
    if target and str(target) in ARCH_FILES:
        return str(target)
    return "DPUCZDX8G_ISA1_B4096"


def architecture_json(target: str) -> Path:
    name = ARCH_FILES.get(target)
    if not name:
        raise ValueError(
            f"Unknown DPU target '{target}'. Supported: {', '.join(ARCH_FILES)}"
        )
    path = _ARCH_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Architecture file missing: {path}")
    return path


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _workspace_needs_ultralytics(workspace: Path) -> bool:
    """Ultralytics is only required when loading a YOLO checkpoint (no --model-class)."""
    meta_path = workspace / "model_meta.json"
    if not meta_path.is_file():
        return False
    import json

    meta = json.loads(meta_path.read_text())
    model_class = meta.get("model_class")
    return model_class is None or model_class == ""


def _stage_ultralytics_patch(workspace: Path) -> None:
    """Copy DPU custom layers and patch script into the Vitis workspace."""
    if not _CUSTOM_LAYERS_DIR.is_dir():
        raise FileNotFoundError(f"Custom layers not found: {_CUSTOM_LAYERS_DIR}")
    if not _PATCH_ULTRALYTICS_SCRIPT.is_file():
        raise FileNotFoundError(f"Patch script not found: {_PATCH_ULTRALYTICS_SCRIPT}")

    layers_dst = workspace / "custom_layers"
    layers_dst.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(_CUSTOM_LAYERS_DIR.glob("*.py")):
        shutil.copy2(src_file, layers_dst / src_file.name)

    scripts_dir = workspace / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_PATCH_ULTRALYTICS_SCRIPT, scripts_dir / "patch_ultralytics.py")


def run_in_vitis_container(
    *,
    workspace: Path,
    image: str,
    command: str,
    vitis_ai_repo: Path | None = None,
    conda_activate: str = CONDA_ACTIVATE_PYTORCH,
    install_ultralytics: bool = False,
) -> None:
    """Run a shell command inside a Vitis AI Docker container."""
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    volumes = [f"{workspace}:/workspace"]
    if vitis_ai_repo and vitis_ai_repo.is_dir():
        volumes.append(f"{vitis_ai_repo.resolve()}:/Vitis-AI")

    setup = conda_activate
    if install_ultralytics:
        setup += f" && pip install -q ultralytics=={VITIS_ULTRALYTICS_VERSION}"
        setup += " && python scripts/patch_ultralytics.py"
    inner = f"{setup} && cd /workspace && {command}"
    cmd = [
        "docker",
        "run",
        "--rm",
        *([item for v in volumes for item in ("-v", v)]),
        "-w",
        "/workspace",
        image,
        "bash",
        "-lc",
        inner,
    ]
    logger.info("Docker: %s", command)
    subprocess.run(cmd, check=True)


def stage_calibration_images(
    workspace: Path,
    calibration_data: Path,
) -> None:
    """Copy calibration images into workspace and write data/val_ids.txt."""
    data_dir = workspace / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cal_images_dir = data_dir / "calibration"
    cal_images_dir.mkdir(parents=True, exist_ok=True)
    cal_dst = data_dir / "val_ids.txt"
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def _collect_image_paths() -> list[Path]:
        if calibration_data.is_file() and calibration_data.suffix == ".txt":
            lines = [
                ln.strip()
                for ln in calibration_data.read_text().splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            paths: list[Path] = []
            list_parent = calibration_data.parent
            for line in lines:
                p = Path(line)
                if not p.is_absolute():
                    p = (list_parent / p).resolve()
                if p.is_file():
                    paths.append(p)
                else:
                    logger.warning("Skipping missing calibration image: %s", p)
            return paths
        if calibration_data.is_dir():
            return sorted(
                p
                for p in calibration_data.iterdir()
                if p.is_file() and p.suffix.lower() in image_suffixes
            )
        raise FileNotFoundError(
            f"calibration_data must be a .txt list or image directory: {calibration_data}"
        )

    sources = _collect_image_paths()
    if not sources:
        raise FileNotFoundError(f"No calibration images found under {calibration_data}")

    rel_paths: list[str] = []
    for i, src in enumerate(sources):
        dst = cal_images_dir / src.name
        if dst.exists():
            dst = cal_images_dir / f"{src.stem}_{i}{src.suffix}"
        shutil.copy2(src, dst)
        rel_paths.append(dst.relative_to(workspace).as_posix())

    cal_dst.write_text("\n".join(rel_paths) + "\n")
    logger.info("Staged %d calibration images under %s", len(rel_paths), cal_images_dir)


def _prepare_pytorch_workspace(
    workspace: Path,
    *,
    model_path: Path,
    calibration_data: Path,
    arch_path: Path,
) -> None:
    scripts_dir = workspace / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    arch_dst = workspace / "architectures"
    arch_dst.mkdir(parents=True, exist_ok=True)

    shutil.copy2(_PYTORCH_SCRIPT, scripts_dir / "vai_q_pytorch.py")
    shutil.copy2(arch_path, arch_dst / arch_path.name)
    stage_calibration_images(workspace, calibration_data)

    workspace = workspace.resolve()
    weights = model_path if model_path.is_absolute() else workspace / model_path.name
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")


def _docker_model_args(workspace: Path) -> str:
    meta_path = workspace / "model_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {meta_path.name} in Vitis workspace; expected staged weights + model class"
        )
    import json

    meta = json.loads(meta_path.read_text())
    model_class = str(meta["model_class"])
    model_kwargs = json.dumps(meta.get("model_kwargs") or {})
    return (
        f"--model_class {shlex.quote(model_class)} "
        f"--model_kwargs {shlex.quote(model_kwargs)}"
    )


def quantize_pytorch(
    workspace: Path,
    *,
    model_filename: str = "model.pt",
    channels: int,
    img_height: int,
    img_width: int,
    dpu_target: str,
    batch_size_calib: int = 16,
    config_file: str | None = None,
    image: str = PYTORCH_IMAGE,
    vitis_ai_repo: Path | None = None,
    pause_seconds: int = 5,
) -> Path:
    """Two-step PTQ: calib then test+deploy. Returns path to quantize_result directory."""
    install_ultralytics = _workspace_needs_ultralytics(workspace)
    if install_ultralytics:
        _stage_ultralytics_patch(workspace)
    script = "python scripts/vai_q_pytorch.py"
    model_arg = f"--model_path {shlex.quote(model_filename)}"
    arch_arg = _docker_model_args(workspace)
    size_args = (
        f"--channels {channels} --img_height {img_height} --img_width {img_width}"
    )
    target_arg = f"--target {shlex.quote(dpu_target)}"
    calib_list = "--calibration_list data/val_ids.txt"
    cfg = f"--config_file {shlex.quote(config_file)}" if config_file else ""

    calib_cmd = (
        f"{script} {model_arg} {arch_arg} --batch_size {batch_size_calib} {size_args} "
        f"{target_arg} --quant_mode calib --inspect {calib_list} {cfg}"
    ).strip()
    run_in_vitis_container(
        workspace=workspace,
        image=image,
        command=calib_cmd,
        vitis_ai_repo=vitis_ai_repo,
        install_ultralytics=install_ultralytics,
    )

    if pause_seconds > 0:
        time.sleep(pause_seconds)

    deploy_cmd = (
        f"{script} {model_arg} {arch_arg} --batch_size 1 {size_args} "
        f"{target_arg} --quant_mode test --deploy {calib_list} {cfg}"
    ).strip()
    run_in_vitis_container(
        workspace=workspace,
        image=image,
        command=deploy_cmd,
        vitis_ai_repo=vitis_ai_repo,
        install_ultralytics=install_ultralytics,
    )

    quant_dir = workspace / "quantize_result"
    if not quant_dir.is_dir():
        raise FileNotFoundError(
            f"quantize_result not found under {workspace} after Vitis-AI export"
        )
    return quant_dir


def find_xmodel(search_dir: Path) -> Path:
    matches = sorted(search_dir.rglob("*.xmodel"))
    if not matches:
        raise FileNotFoundError(f"No .xmodel under {search_dir}")
    return matches[0]


def compile_xmodel(
    workspace: Path,
    xmodel_path: Path,
    arch_path: Path,
    *,
    output_subdir: str = "compiled",
    model_name: str = "deployed",
    image: str = PYTORCH_IMAGE,
    vitis_ai_repo: Path | None = None,
) -> Path:
    """Compile xmodel for DPU using vai_c_xir inside Docker."""
    rel_xmodel = xmodel_path.relative_to(workspace)
    rel_arch = arch_path.relative_to(workspace)
    out_dir = workspace / output_subdir
    compile_cmd = (
        f"vai_c_xir -x {shlex.quote(str(rel_xmodel))} "
        f"-a {shlex.quote(str(rel_arch))} "
        f"-o {shlex.quote(output_subdir)} -n {shlex.quote(model_name)}"
    )
    run_in_vitis_container(
        workspace=workspace,
        image=image,
        command=compile_cmd,
        vitis_ai_repo=vitis_ai_repo,
    )
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Compiler output missing: {out_dir}")
    return out_dir


def export_pytorch_to_dpu(
    workspace: Path,
    model_path: Path,
    calibration_data: Path,
    *,
    channels: int,
    height: int,
    width: int,
    dpu_target: str,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    """Full flow: quantize → xmodel → vai_c_xir compile."""
    workspace = workspace.resolve()
    if not _docker_available():
        raise RuntimeError(
            "Docker is required for Vitis-AI export (install Docker and pull Vitis-AI image)."
        )

    vitis_repo = resolve_vitis_ai_repo(config)
    image = str(config.get("vitis_ai_image", PYTORCH_IMAGE))
    arch_src = architecture_json(dpu_target)
    arch_in_workspace = workspace / "architectures" / arch_src.name
    repo_mount = vitis_repo if vitis_repo.is_dir() else None

    _prepare_pytorch_workspace(
        workspace,
        model_path=model_path,
        calibration_data=calibration_data,
        arch_path=arch_src,
    )

    quant_dir = quantize_pytorch(
        workspace,
        model_filename=model_path.name,
        channels=channels,
        img_height=height,
        img_width=width,
        dpu_target=dpu_target,
        batch_size_calib=int(config.get("calib_batch_size", 16)),
        config_file=config.get("quant_config_file"),
        image=image,
        vitis_ai_repo=repo_mount,
        pause_seconds=int(config.get("quant_pause_seconds", 5)),
    )

    xmodel = find_xmodel(quant_dir)
    compiled = compile_xmodel(
        workspace,
        xmodel,
        arch_in_workspace,
        output_subdir=str(config.get("compile_output_dir", "compiled")),
        model_name=str(config.get("compile_model_name", "deployed")),
        image=image,
        vitis_ai_repo=repo_mount,
    )
    return xmodel, compiled
