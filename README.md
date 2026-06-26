# Edge AI Deployment Pipeline (Computer Vision)
#ReMi Research Group HSBI

Modular pipeline for **computer-vision models** — profile, check compatibility, rank boards, choose a target, and deploy.

**Supported host OS:** **Linux only** (Ubuntu 22.04+ recommended).

Requires **Python 3.11+**.

## Pipeline

End-to-end flow (all stages run in order):

| Stage | Purpose |
|-------|---------|
| **Intake** | Load Keras (`.keras`/`.h5`) or PyTorch **weights** (`.pt`/`.pth`) + `--model-class` |
| **Profiler** | FLOPs, parameter count, memory estimates, layer inventory, CPU latency |
| **Compatibility** | Match layer types against each board's toolchain allowlist/denylist |
| **Constraint** | Score viable boards on power, cost, and memory; rank candidates |
| **Board selection** | Pick a ranked board (interactive prompt, `--board-id`, or `--auto-select`) |
| **Deployment** | Emit a deployable artifact for the selected target |

On success the CLI prints a **JSON summary** to stdout (including `rankings` and the selected board) and writes deployment artifacts under `-o` (default `data/artifacts/`).

Profiling, compatibility, and constraint checking run on CPU. **Deployment** may require extra host tools (Docker, CUDA, STEdgeAI, etc.) depending on the target platform.

## Supported platforms

Configured in bundled `src/edge_ai_deployment_pipeline/config/platforms.json`:

| Family | Board | Toolchain | Latency type |
|--------|-------|-----------|--------------|
| Jetson | `jetson_nano`, `agx_orin` | TensorRT | `medium` |
| FPGA | `zcu104` | Vitis AI | `ultra_low` |
| MCU | `stm32n6570_dk` | STM32N6 | `low` |
| ARM | `raspberry_pi_4` | ncnn | `moderate` |

Override with `--platform-config path/to/platforms.json`. Each board must declare `available_toolchains`.

## Model formats

### PyTorch (all platforms that accept PyTorch)

PyTorch intake is **weights + architecture only**:

1. Define `nn.Module` in an importable `.py` file.
2. Save **weights only**: `torch.save(model.state_dict(), "weights.pt")`.
3. Pass `--model-class file.py:ClassName` to the CLI.

Do **not** use `torch.save(model, path)` (full-model pickle). It is not supported.

`--model-class` formats:

- `data/model_class/cifar_net.py:Net` — path to a `.py` file
- `mypackage.models:Net` — importable Python module

Optional constructor args: `--model-kwargs '{"num_classes": 10}'`.

**Example network** (`data/model_class/cifar_net.py`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

**After training:**

```python
net = Net()
# ... train ...
torch.save(net.state_dict(), "cifar10.pt")
```

### Keras (Jetson, ARM, MCU)

Load a full saved model (architecture + weights): `.keras` or `.h5`.

Weights-only `.h5` files are not supported. MCU requires Keras; FPGA requires PyTorch.

### Per-target summary

| Target | Accepted at deploy | Notes |
|--------|-------------------|-------|
| **FPGA** (`zcu104`) | PyTorch weights + `--model-class` | Vitis-AI Docker PTQ |
| **Jetson** | PyTorch weights or Keras | PyTorch → ONNX → TensorRT; Keras via tf2onnx |
| **ARM** (`raspberry_pi_4`) | PyTorch weights or Keras | PyTorch → ONNX → ncnn |
| **MCU** (`stm32n6570_dk`) | Keras `.keras` / `.h5` only | TFLite + STEdgeAI |

Use `--latency-type` that matches your intended deployment family so constraint ranking considers the right boards.

Framework mismatches (e.g. Keras → FPGA) are **not** flagged during compatibility; they fail at **deployment** with a clear error.

## Linux setup

### 1. System prerequisites

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Optional but required for specific deploy targets:
#   FPGA  → Docker
#   Jetson → NVIDIA driver + CUDA (build TensorRT engines on a CUDA machine)
#   MCU   → STMicroelectronics STEdgeAI (Linux installer)
sudo apt install -y docker.io
sudo usermod -aG docker "$USER"   # log out/in so docker runs without sudo
```

### 2. Clone and create a virtual environment

```bash
git clone https://github.com/surajkarki66/edge_ai_deployment_pipeline.git
cd edge_ai_deployment_pipeline

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Conda works too (`conda create -n edge-ai python=3.11`).

### 3. Install Python dependencies

**Option A — editable install (recommended):**

```bash
pip install -e .
```

This installs runtime deps from `pyproject.toml` (PyTorch, TensorFlow, ONNX stack, ncnn, TensorRT pin, etc.).

**Option B — pinned requirements file:**

`requirements.txt` pins a CUDA-enabled PyTorch wheel. Install PyTorch from the official index first, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
```

For **CPU-only** profiling (no Jetson TensorRT build), a plain `pip install -e .` is usually enough; TensorRT engine build will fail later without CUDA, which is expected.

Optional extras:

```bash
pip install -e ".[dev]"   # mypy, ruff, pre-commit
pip install -e ".[mcu]"   # adds mako (STM32N6 compile)
```

### 4. Calibration data

INT8 / PTQ toolchains need images under `calibration_data` in `platforms.json` (default: `data/calibration/`). Point `toolchain_overrides` at your dataset if needed:

- A **directory** of `.jpg` / `.png` images, or
- A **`.txt` file** listing image paths (one per line)

Bundled sample images live in `data/calibration/` for smoke tests.

### 5. Platform-specific host tools

| Target | Host requirements |
|--------|-------------------|
| **FPGA** | [Docker](https://docs.docker.com/engine/install/) + `docker pull xilinx/vitis-ai-pytorch-cpu:latest`. Optional: clone [Vitis-AI](https://github.com/Xilinx/Vitis-AI) and set `VITIS_AI_REPO`. |
| **Jetson** | NVIDIA GPU with CUDA; TensorRT **10.x** (`tensorrt>=10.16.1.11,<11`) for INT8 calibration. |
| **ARM** | ncnn + pnnx (installed via pip). |
| **MCU** | [STEdgeAI](https://www.st.com/en/embedded-software/x-cube-ai.html) under `/opt/ST/STEdgeAI/<version>/Utilities/linux/stedgeai`. Set `STM32_AI_EXE`. `mako` required for compile. |

```bash
docker pull xilinx/vitis-ai-pytorch-cpu:latest
export STM32_AI_EXE=/opt/ST/STEdgeAI/4.0/Utilities/linux/stedgeai
```

### 6. Verify installation

```bash
edge-ai-deploy --help
python -c "import torch, tensorflow; print('torch', torch.__version__, 'tf', tensorflow.__version__)"
```

## Run

### PyTorch → FPGA (CIFAR-10 example)

```bash
edge-ai-deploy data/cifar10.pt \
  --model-class data/model_class/cifar_net.py:Net \
  --input-shape 3,32,32 \
  --latency-type ultra_low \
  --max-cost-usd 2000 \
  --auto-select
```

### PyTorch → ARM / Jetson

```bash
edge-ai-deploy data/cifar10.pt \
  --model-class data/model_class/cifar_net.py:Net \
  --input-shape 3,32,32 \
  --latency-type moderate \
  --auto-select
```

### Keras → MCU

```bash
edge-ai-deploy path/to/model.keras \
  --input-shape 3,96,96 \
  --latency-type low \
  --board-id stm32n6570_dk \
  --auto-select
```

`--input-shape` is **required** (`C,H,W`, e.g. `3,224,224` for RGB).

`--latency-type` is **required**:

| Value | Typical platforms |
|-------|-------------------|
| `ultra_low` | FPGA (`zcu104`) |
| `low` | MCU (`stm32n6570_dk`) |
| `medium` | NVIDIA Jetson (`jetson_nano`, `agx_orin`) |
| `moderate` | Raspberry Pi (`raspberry_pi_4`) |

Or invoke the module directly:

```bash
python -m edge_ai_deployment_pipeline.main data/cifar10.pt \
  --model-class data/model_class/cifar_net.py:Net \
  --input-shape 3,32,32 \
  --latency-type ultra_low \
  --auto-select
```

### Board selection

After constraint ranking, the CLI either:

1. **Prompts** you to pick a rank (default), or
2. Uses **`--board-id zcu104`** to deploy to a specific board, or
3. Uses **`--auto-select`** to take rank #1 without prompting.

Example with budget filters:

```bash
edge-ai-deploy data/cifar10.pt \
  --model-class data/model_class/cifar_net.py:Net \
  --input-shape 3,32,32 \
  --latency-type ultra_low \
  --max-power-w 15 \
  --max-cost-usd 2000 \
  --auto-select
```

JSON output includes `rankings` (all scored boards) and `selected_platform` (the board used for deployment).

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `model` | — | **Required.** `.pt`/`.pth` weights or `.keras`/`.h5` model |
| `--latency-type` | — | **Required.** `ultra_low`, `low`, `medium`, or `moderate` |
| `--input-shape` | — | **Required.** `C,H,W` (e.g. `3,224,224`) |
| `--model-class` | — | **Required for `.pt`/`.pth`.** e.g. `data/model_class/cifar_net.py:Net` |
| `--model-kwargs` | — | JSON constructor kwargs, e.g. `'{"num_classes": 10}'` |
| `-o`, `--output-dir` | `data/artifacts` | Directory for deployment artifacts |
| `--platform-config` | bundled `platforms.json` | Custom platform catalogue |
| `--max-power-w` | `25` | Boards above this TDP are **excluded** |
| `--max-cost-usd` | `500` | Boards above this cost are **excluded** |
| `--toolchain` | — | Force a toolchain (`tensorrt`, `vitis_ai`, `ncnn`, `stm32n6`, …) |
| `--board-id` | — | Deploy to this board (skip interactive pick) |
| `--auto-select` | off | Use highest-ranked board without prompting |
| `-v`, `--verbose` | off | Debug logging |

## Constraint scoring

For each viable board matching `--latency-type`, the best toolchain per board is scored and all candidates are **ranked** (highest score first).

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

## Deployment notes

The **deployment** stage runs the platform adapter for the selected board. Expect target-specific failures if host tools are missing.

| Target | Output | Notes |
|--------|--------|-------|
| **FPGA** | `.xmodel` | Vitis-AI PyTorch PTQ + DPU compile in Docker. Stages weights and architecture into `data/artifacts/<board>/vitis_workspace/`. |
| **Jetson** | TensorRT `.engine` | PyTorch → ONNX → TensorRT. Requires CUDA. INT8 needs `calibration_data` and TensorRT 10.x. |
| **ARM** | ncnn `.param` / `.bin` | PyTorch → ONNX → pnnx. Keras via tf2onnx → onnx2torch. |
| **MCU** | STEdgeAI bundle | Keras → TFLite → `stedgeai` compile. Requires `mako`. |

Artifacts are written under `<output-dir>/<board-id>/`.

Logs are written under `logs/` (one file per run).

## Project layout

```
edge_ai_deployment_pipeline/
├── src/edge_ai_deployment_pipeline/
│   ├── main.py                 # CLI entry (edge-ai-deploy)
│   ├── core/                   # pipeline, CLI, pytorch_loader, board selection
│   ├── stages/                 # intake, profiler, compatibility, constraint, deployment
│   ├── adapters/               # jetson, fpga, mcu, arm toolchains
│   └── config/platforms.json   # platform catalogue
├── data/                       # centralized data root
│   ├── model_class/cifar_net.py  # example PyTorch network definition
│   ├── calibration/            # sample calibration images
│   └── artifacts/              # deployment output (created at run time)
├── logs/                       # pipeline logs (created at run time)
├── pyproject.toml
├── requirements.txt
└── ruff.toml
```
