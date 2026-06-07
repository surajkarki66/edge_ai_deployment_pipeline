# Edge AI Deployment Pipeline (Computer Vision)

Modular pipeline for **computer-vision models** — profile, check compatibility, rank boards, and deploy to the selected edge target.

Requires **Python 3.11+**.

## Pipeline

End-to-end flow (all stages run in order):

| Stage | Purpose |
|-------|---------|
| **Intake** | Load model from disk (PyTorch or Keras) |
| **Profiler** | FLOPs, parameter count, memory estimates, layer inventory, CPU latency |
| **Compatibility** | Match layer types against each board's toolchain allowlist/denylist |
| **Constraint** | Score viable boards on power, cost, and memory; pick best board × toolchain |
| **Deployment** | Emit a deployable artifact for the selected target |

On success the CLI prints a **JSON summary** to stdout and writes deployment artifacts under `-o` (default `artifacts/`).

## Layout

```
edge_ai_deployment_pipeline/          # project root
├── src/edge_ai_deployment_pipeline/
│   ├── main.py                       # CLI entry point
│   ├── core/                         # pipeline, CLI, types
│   ├── stages/                       # intake, profiler, compatibility, constraint, deployment
│   ├── adapters/                     # platform toolchains (jetson, fpga, mcu, arm)
│   └── config/                       # platforms.json
├── setup.py
├── pyproject.toml
├── requirements.txt
└── ruff.toml
```

## Install

Minimal install (pulls all runtime deps from `pyproject.toml`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or install pinned deps from `requirements.txt` first, then the package:

```bash
pip install -r requirements.txt
pip install -e .
```

Optional extras (install only what you need):

```bash
pip install -e ".[pytorch]"      # torch, fvcore
pip install -e ".[tensorflow]"  # tensorflow
pip install -e ".[arm]"          # ncnn, pnnx
pip install -e ".[jetson]"        # TensorRT / ONNX stack (includes tf2onnx)
pip install -e ".[dev]"           # mypy, ruff, pre-commit
```

## Run

```bash
edge-ai-deploy path/to/model.pt --input-shape 3,224,224 --latency-type ultra_low -o artifacts/
```

`--input-shape` is **required** (`C,H,W`, e.g. `3,224,224`).

`--latency-type` is **required**. Pick the latency class for your deployment target:

| Value | Typical platforms |
|-------|-------------------|
| `ultra_low` | FPGA (`zcu104`) |
| `low` | MCU (`stm32n6570_dk`) |
| `medium` | NVIDIA Jetson (`jetson_nano`, `agx_orin`) |
| `moderate` | Raspberry Pi (`raspberry_pi_4`) |

Supported model formats: `.pt`, `.pth`, `.keras`, `.h5`.

Or invoke the module directly:

```bash
python -m edge_ai_deployment_pipeline.main path/to/model.pt --input-shape 3,224,224 --latency-type low
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--latency-type` | — | **Required.** `ultra_low`, `low`, `medium`, or `moderate` (see table above) |
| `-o`, `--output-dir` | `artifacts` | Directory for deployment artifacts |
| `--platform-config` | bundled `platforms.json` | Custom platform catalogue |
| `--max-power-w` | `25` | Max board power budget (W) |
| `--max-cost-usd` | `500` | Max board cost budget (USD) |
| `--toolchain` | — | Force a toolchain (`tensorrt`, `vitis_ai`, `ncnn`, `stm32n6`, …) |
| `-v`, `--verbose` | off | Debug logging |

## Constraint scoring

For each viable board matching `--latency-type`, the best toolchain is scored and the highest total wins.

**Score weights:** power 50%, cost 50%.

**Memory check:** `model_size_mb + max(peak_activation_memory_mb, inference_ram_mb)` must fit in the board's `memory_mb`. If not, a **−50** penalty is applied.

Boards whose `tdp_watts` or `cost_usd` exceed `--max-power-w` or `--max-cost-usd` are **excluded**. The pipeline exits with an error listing each rejected board and why.

## Compatibility

Each toolchain uses either an **allowlist** (FPGA Vitis AI, MCU STM32N6) or **denylist** (Jetson TensorRT, ARM ncnn) of layer/operation names.

| Status | Meaning |
|--------|---------|
| `fully_compatible` | No unsupported operations |
| `partial` | Some unsupported ops, but ≤ 50% of layer inventory |
| `incompatible` | > 50% unsupported — board is excluded |

Partial boards remain viable but are noted in constraint reasoning.

## Supported platforms

Configured in bundled `config/platforms.json`:

| Family | Board | Toolchain | Latency type |
|--------|-------|-----------|--------------|
| Jetson | `jetson_nano`, `agx_orin` | TensorRT | `medium` |
| FPGA | `zcu104` | Vitis AI | `ultra_low` |
| MCU | `stm32n6570_dk` | STM32N6 | `low` |
| ARM | `raspberry_pi_4` | ncnn | `moderate` |

Override with `--platform-config path/to/platforms.json`. Each board must declare `available_toolchains`.

## Deployment notes

The **deployment** stage invokes the platform adapter for the winning board. That requires the corresponding toolchain to be installed and configured on the host (Vitis AI Docker image, TensorRT, ncnn, STM32N6 SDK, etc.).

Quantized toolchains need a valid `calibration_data` path in `platforms.json` — update board `toolchain_overrides` to point at your calibration dataset before running a full deploy.
