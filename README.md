# Edge AI Deployment Pipeline (Computer Vision)

Modular pipeline for **computer-vision models** — profile, check compatibility, rank boards, choose a target, and deploy.

**Supported host OS:** **Linux only** (Ubuntu 22.04+ recommended).

Requires **Python 3.11+**.

## Pipeline

End-to-end flow (all stages run in order):


| Stage               | Purpose                                                                    |
| ------------------- | -------------------------------------------------------------------------- |
| **Intake**          | Load model from disk (PyTorch or Keras)                                    |
| **Profiler**        | FLOPs, parameter count, memory estimates, layer inventory, CPU latency     |
| **Compatibility**   | Match layer types against each board's toolchain allowlist/denylist        |
| **Constraint**      | Score viable boards on power, cost, and memory; rank candidates            |
| **Board selection** | Pick a ranked board (interactive prompt, `--board-id`, or `--auto-select`) |
| **Deployment**      | Emit a deployable artifact for the selected target                         |


On success the CLI prints a **JSON summary** to stdout (including `rankings` and the selected board) and writes deployment artifacts under `-o` (default `artifacts/`).

Profiling, compatibility, and constraint checking run on CPU. **Deployment** may require extra host tools (Docker, CUDA, STEdgeAI, etc.) depending on the target platform.

## Supported platforms

Configured in bundled `src/edge_ai_deployment_pipeline/config/platforms.json`:


| Family | Board                     | Toolchain | Latency type |
| ------ | ------------------------- | --------- | ------------ |
| Jetson | `jetson_nano`, `agx_orin` | TensorRT  | `medium`     |
| FPGA   | `zcu104`                  | Vitis AI  | `ultra_low`  |
| MCU    | `stm32n6570_dk`           | STM32N6   | `low`        |
| ARM    | `raspberry_pi_4`          | ncnn      | `moderate`   |


Override with `--platform-config path/to/platforms.json`. Each board must declare `available_toolchains`.

## Model formats

**Intake** accepts `.pt`, `.pth`, `.keras`, and `.h5`. **Deployment** enforces per-platform framework rules (checked at deploy time, not during compatibility):


| Target                     | Model format              | Notes                                                             |
| -------------------------- | ------------------------- | ----------------------------------------------------------------- |
| **FPGA** (`zcu104`)        | PyTorch `.pt` / `.pth`    | Eager `nn.Module` only                                            |
| **MCU** (`stm32n6570_dk`)  | Keras `.keras` / `.h5`    | Full saved model (architecture + weights), not weights-only `.h5` |
| **Jetson**                 | `.pt` or `.keras` / `.h5` | Keras path uses tf2onnx → ONNX → TensorRT                         |
| **ARM** (`raspberry_pi_4`) | `.pt` or `.keras` / `.h5` | Keras path uses tf2onnx → onnx2torch → ncnn                       |


Use `--latency-type` that matches your intended deployment family so constraint ranking considers the right boards.

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
sudo apt install -y docker.io    # FPGA Vitis-AI export
sudo usermod -aG docker "$USER"  # log out/in so docker runs without sudo
```

### 2. Clone and create a virtual environment

```bash
git https://github.com/surajkarki66/edge_ai_deployment_pipeline.git
cd edge_ai_deployment_pipeline

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Conda** works too (`conda create -n edge-ai python=3.11`).

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

Optional extras if you want a slimmer base install later:

```bash
pip install -e ".[dev]"        # mypy, ruff, pre-commit
pip install -e ".[mcu]"        # adds mako (STM32N6 compile)
```

### 4. Calibration data

INT8 / PTQ toolchains need images under `calibration_data` in `platforms.json` (default: `data/calibration/`). Point `toolchain_overrides` at your dataset if needed:

- A **directory** of `.jpg` / `.png` images, or
- A `**.txt` file** listing image paths (one per line)

Bundled sample images live in `data/calibration/` for smoke tests.

### 5. Platform-specific host tools

Install only what you need for your target board:


| Target     | Host requirements                                                                                                                                                                                                                   |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FPGA**   | [Docker](https://docs.docker.com/engine/install/) + pull `xilinx/vitis-ai-pytorch-cpu:latest`. Optional: clone [Vitis-AI](https://github.com/Xilinx/Vitis-AI) and set `VITIS_AI_REPO` if you rely on extra repo assets.             |
| **Jetson** | NVIDIA GPU with CUDA on the build machine; TensorRT **10.x** (`tensorrt>=10.16.1.11,<11`) for INT8 calibration. Engines are typically built on the Jetson or another CUDA host.                                                     |
| **ARM**    | ncnn + pnnx (installed via pip). No extra daemon.                                                                                                                                                                                   |
| **MCU**    | [STEdgeAI](https://www.st.com/en/embedded-software/x-cube-ai.html) Linux package under `/opt/ST/STEdgeAI/<version>/Utilities/linux/stedgeai`. Set `STM32_AI_EXE` or pass path via toolchain config. `mako` is required for compile. |


**FPGA Docker pull (one-time):**

```bash
docker pull xilinx/vitis-ai-pytorch-cpu:latest
```

**MCU STEdgeAI (example):**

```bash
export STM32_AI_EXE=/opt/ST/STEdgeAI/4.0/Utilities/linux/stedgeai
```

### 6. Verify installation

```bash
edge-ai-deploy --help
python -c "import torch, tensorflow; print('torch', torch.__version__, 'tf', tensorflow.__version__)"
```

## Run

Basic example (FPGA / PyTorch):

```bash
edge-ai-deploy path/to/model.pt \
  --input-shape 3,224,224 \
  --latency-type ultra_low \
  --board-id zcu104 \
  --auto-select \
  -o artifacts/
```

MCU / Keras example:

```bash
edge-ai-deploy path/to/model.keras \
  --input-shape 3,96,96 \
  --latency-type low \
  --board-id stm32n6570_dk \
  --auto-select \
  -o artifacts/
```

`--input-shape` is **required** (`C,H,W`, e.g. `3,224,224` for RGB).

`--latency-type` is **required**:


| Value       | Typical platforms                         |
| ----------- | ----------------------------------------- |
| `ultra_low` | FPGA (`zcu104`)                           |
| `low`       | MCU (`stm32n6570_dk`)                     |
| `medium`    | NVIDIA Jetson (`jetson_nano`, `agx_orin`) |
| `moderate`  | Raspberry Pi (`raspberry_pi_4`)           |


Or invoke the module directly:

```bash
python -m edge_ai_deployment_pipeline.main path/to/model.pt \
  --input-shape 3,224,224 \
  --latency-type medium
```

### Board selection

After constraint ranking, the CLI either:

1. **Prompts** you to pick a rank (default), or
2. Uses `**--board-id zcu104`** to deploy to a specific board, or
3. Uses `**--auto-select**` to take rank #1 without prompting.

Example with interactive selection omitted:

```bash
edge-ai-deploy model.pt --input-shape 3,224,224 --latency-type medium \
  --max-power-w 15 --max-cost-usd 150 --auto-select
```

JSON output includes `rankings` (all scored boards) and `selected_platform` (the board used for deployment).

### CLI options


| Flag                 | Default                  | Description                                                      |
| -------------------- | ------------------------ | ---------------------------------------------------------------- |
| `--latency-type`     | —                        | **Required.** `ultra_low`, `low`, `medium`, or `moderate`        |
| `--input-shape`      | —                        | **Required.** `C,H,W` (e.g. `3,224,224`)                         |
| `-o`, `--output-dir` | `artifacts`              | Directory for deployment artifacts                               |
| `--platform-config`  | bundled `platforms.json` | Custom platform catalogue                                        |
| `--max-power-w`      | `25`                     | Boards above this TDP are **excluded**                           |
| `--max-cost-usd`     | `500`                    | Boards above this cost are **excluded**                          |
| `--toolchain`        | —                        | Force a toolchain (`tensorrt`, `vitis_ai`, `ncnn`, `stm32n6`, …) |
| `--board-id`         | —                        | Deploy to this board (skip interactive pick)                     |
| `--auto-select`      | off                      | Use highest-ranked board without prompting                       |
| `-v`, `--verbose`    | off                      | Debug logging                                                    |


## Constraint scoring

For each viable board matching `--latency-type`, the best toolchain per board is scored and all candidates are **ranked** (highest score first).

**Score weights:** power 50%, cost 50%.

**Memory check:** `model_size_mb + max(peak_activation_memory_mb, inference_ram_mb)` must fit in the board's `memory_mb`. If not, a **−50** penalty is applied.

Boards whose `tdp_watts` or `cost_usd` exceed `--max-power-w` or `--max-cost-usd` are **excluded**. The pipeline exits with an error listing each rejected board and why.

## Compatibility

Each toolchain uses either an **allowlist** (FPGA Vitis AI, MCU STM32N6) or **denylist** (Jetson TensorRT, ARM ncnn) of layer/operation names.


| Status             | Meaning                                            |
| ------------------ | -------------------------------------------------- |
| `fully_compatible` | No unsupported operations                          |
| `partial`          | Some unsupported ops, but ≤ 50% of layer inventory |
| `incompatible`     | > 50% unsupported — board is excluded              |


Partial boards remain viable but are noted in constraint reasoning. Framework mismatches (e.g. Keras model → FPGA) are **not** flagged here; they fail at **deployment** with a clear error.

## Deployment notes

The **deployment** stage runs the platform adapter for the selected board. Expect target-specific failures if host tools are missing.

- **FPGA:** Runs Vitis-AI PyTorch quantization and DPU compile inside Docker. First run pulls a large image. Model must be eager PyTorch, not TorchScript.
- **Jetson:** Builds a TensorRT engine via ONNX. Requires CUDA on the host for engine build. INT8 needs `calibration_data` and TensorRT 10.x.
- **ARM:** Exports ncnn `.param` / `.bin` via PyTorch → ONNX → pnnx (Keras models round-trip through tf2onnx and onnx2torch).
- **MCU:** Quantizes Keras to TFLite, then compiles with STEdgeAI. Requires the Linux `stedgeai` binary and `mako`.

Artifacts are written under `<output-dir>/<board-id>/`.

## Project layout

```
edge_ai_deployment_pipeline/
├── src/edge_ai_deployment_pipeline/
│   ├── main.py                       # CLI entry (edge-ai-deploy)
│   ├── core/                         # pipeline, CLI, board selection, types
│   ├── stages/                       # intake, profiler, compatibility, constraint, deployment
│   ├── adapters/                     # jetson, fpga, mcu, arm toolchains
│   └── config/platforms.json         # platform catalogue
├── data/calibration/                 # sample calibration images
├── pyproject.toml
├── requirements.txt
└── ruff.toml
```

