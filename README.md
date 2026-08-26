# Edge AI Deployment Pipeline for Computer Vision Applications

**Resource-Efficient Microelectronics and Cognitive Edge Computing (ReMi) Research Group — HSBI**

**Supported host OS:** **Linux only**.

Requires **Python 3.11+**.

## Pipeline

End-to-end flow (all steps should be run in order):

| Step                      | Purpose                                                                                                                                                           |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Intake**          | Load Keras (`.keras`/`.h5`), PyTorch **weights** (`.pt`/`.pth`) + `--model-class`, or Ultralytics YOLO (`.pt`) with `--task object_detection` |
| **Profiler**        | FLOPs, parameter count, memory estimates, layer inventory, CPU latency                                                                                            |
| **Compatibility**   | Match layer types against each board's toolchain allowlist/denylist                                                                                               |
| **Constraint**      | Score viable boards on power, cost, and memory; rank candidates                                                                                                   |
| **Board selection** | Pick a ranked board (interactive prompt,`--board-id`, or `--auto-select`)                                                                                     |
| **Deployment**      | Emit a deployable artifact for the selected target                                                                                                                |

On success, the CLI prints a **JSON summary** to stdout (including `rankings`, `run_id`, and the selected board) and writes deployment artifacts under `data/artifacts/<run-id>/<board-id>/` (override base path with `-o`).

Profiling, compatibility, and constraint checking run on the CPU. **Deployment** requires the target-specific host tools installed in [Linux setup §5](#5-platform-specific-host-tools) (Docker, CUDA, STEdgeAI, etc.).

## Supported platforms

Configured in bundled `src/edge_ai_deployment_pipeline/config/platforms.json`:

| Family | Board                         | Toolchain | Latency type |
| ------ | ----------------------------- | --------- | ------------ |
| Jetson | `jetson_nano`, `agx_orin` | TensorRT  | `medium`   |

Override with `--platform-config path/to/platforms.json`. Each board must declare `available_toolchains`.

## Model formats

| Model format                                              | `--task`               | Jetson | ARM | FPGA | MCU |
| --------------------------------------------------------- | ------------------------ | ------ | --- | ---- | --- |
| PyTorch weights (`.pt`/`.pth` + `--model-class`)    | `image_classification` | Yes    | Yes | Yes  | No  |
| Keras full model (`.keras` / `.h5`)                   | `image_classification` | Yes    | Yes | No   | Yes |
| Ultralytics YOLO (`.pt`/`.pth`, no `--model-class`) | `object_detection`     | Yes    | Yes | Yes  | Yes |

Keras requires a **full saved model** (architecture + weights). Weights-only `.h5` files are not supported. PyTorch intake is **weights + architecture only**.

### PyTorch

PyTorch intake is **weights + architecture only**:

1. Define `nn.Module` in an importable `.py` file.
2. Save **weights only**: `torch.save(model.state_dict(), "weights.pt")`.
3. Pass `--model-class file.py:ClassName` to the CLI.

Do **not** use full-model pickle. It is not supported.

`--model-class` formats:

- `data/model_class/mnist_net.py:Net` — path to a `.py` file
- `mypackage.models:Net` — importable Python module

Optional constructor args: `--model-kwargs '{"num_classes": 10}'`.

**Example network** (`data/model_class/mnist_net.py`):

```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(num_features=16)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(num_features=16)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.drop1 = nn.Dropout2d(p=0.1)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    self.bn3 = nn.BatchNorm2d(num_features=32)
    self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
    self.bn4 = nn.BatchNorm2d(num_features=32)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.drop2 = nn.Dropout2d(p=0.1)
    self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=0)
    self.bn5 = nn.BatchNorm2d(num_features=16)
    self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
    self.bn6 = nn.BatchNorm2d(num_features=16)
    self.gap1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.fc1 = nn.Linear(in_features=16, out_features=10)

    def forward(self, x):
    x = self.drop1(self.pool1(self.bn2(F.relu(self.conv2(self.bn1(F.relu(self.conv1(x))))))))
    x = self.drop2(self.pool2(self.bn4(F.relu(self.conv4(self.bn3(F.relu(self.conv3(x))))))))
    x = self.bn6(F.relu(self.conv6(self.bn5(F.relu(self.conv5(x))))))
    x = self.gap1(x)
    x = x.view(-1, 16)
    x = self.fc1(x)
    return F.log_softmax(x, dim=1)
```

**After training:**

```python
net = Net()
# ... train ...
torch.save(net.state_dict(), "mnist_model.pt")
```

### Keras

Load a full saved model (architecture + weights): `.keras` or `.h5` (see table above).

### Ultralytics YOLO

For `--task object_detection`, pass an Ultralytics checkpoint (`.pt` or `.pth`) **without** `--model-class`. The pipeline loads it via the `ultralytics` package (pinned to **8.4.24** in `pyproject.toml`).

| Target                     | Host load behaviour                                                                                                                                                                                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **FPGA**             | When you pass`--board-id zcu104` or `--toolchain vitis_ai`, the pipeline patches `ultralytics/nn/modules/head.py` and `block.py` with DPU-compatible layers **before** loading (same patch applied again inside the Vitis-AI Docker container before PTQ). |
| **MCU, Jetson, ARM** | Stock Ultralytics modules — no DPU patch on the host.                                                                                                                                                                                                                   |

**FPGA: train with DPU-compatible forks first.** Stock Ultralytics checkpoints for **YOLOv8**, **YOLOv11**, and **YOLOv26** cannot be compiled to FPGA directly — the Vitis AI DPU toolchain reports **unsupported layers** during compatibility. Retrain (or fine-tune) with the forks below, then deploy the resulting `.pt` checkpoint through this pipeline:

| YOLO variant     | Training repository                                                                    |
| ---------------- | -------------------------------------------------------------------------------------- |
| YOLOv8           | [surajkarki66/yolov8-ultralytics](https://github.com/surajkarki66/yolov8-ultralytics)   |
| YOLOv11, YOLOv26 | [surajkarki66/yolov26-ultralytics](https://github.com/surajkarki66/yolov26-ultralytics) |

The host-side DPU patch (table above) adjusts the detection head for export; training with these repositories ensures the **full network** is DPU-compatible before you run `edge-ai-deploy`.

Custom DPU layers live under `src/edge_ai_deployment_pipeline/adapters/fpga/toolchains/ultralytics_dpu/custom_layers/`.

Use `--latency-type` that matches your intended deployment family so constraint ranking considers the right boards.

## Linux setup

### 1. System prerequisites

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git
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

```bash
pip install -e .
```

This installs runtime deps from `pyproject.toml` (PyTorch, TensorFlow, ONNX stack, ncnn, TensorRT, `mako`, etc.) — everything needed for end-to-end deployment.

### 4. Calibration data

INT8 / PTQ toolchains need images under `calibration_data` in `platforms.json` (default: `data/calibration/`). Point `toolchain_overrides` at your dataset if needed:

- A **directory** of `.jpg` / `.png` images, or
- A `**.txt` file** listing image paths (one per line)

Bundled sample images live in `data/calibration/` for PTQ examples.

### 5. Platform-specific host tools

This pipeline is **end-to-end** — all steps through deployment must succeed. Install the tools for the board family you deploy to:

| Target                             | Host tools                            | Environment variables                    |
| ---------------------------------- | ------------------------------------- | ---------------------------------------- |
| **FPGA** (`zcu104`)        | Docker + Vitis-AI image               | —                                       |
| **Jetson**                   | NVIDIA driver, CUDA, TensorRT 10.x    | —                                       |
| **ARM** (`raspberry_pi_4`) | ncnn + pnnx (via`pip install -e .`) | —                                       |
| **MCU** (`stm32n6570_dk`)  | STEdgeAI, STM32CubeIDE                | `STM32_AI_EXE`, `STM32_CUBE_IDE_EXE` |

#### FPGA — Docker + Vitis AI

This pipeline deploys to FPGA (`zcu104`) via **AMD Vitis AI** (DPU PTQ + compile). Required host setup:

```bash
sudo apt install -y docker.io
sudo usermod -aG docker "$USER"   # log out/in so docker runs without sudo

docker pull xilinx/vitis-ai-pytorch-cpu:latest
```

Optional: clone [Vitis-AI](https://github.com/Xilinx/Vitis-AI) and set `VITIS_AI_REPO`. Ultralytics **8.4.24** is installed with this package.

**Alternative: FINN+ compiler.** To deploy quantized neural networks on FPGA with the **FINN+** dataflow compiler instead of Vitis AI, use the separate [surajkarki66/finn-plus](https://github.com/surajkarki66/finn-plus) repository — it is not invoked by `edge-ai-deploy`. FINN+ targets QNN inference on FPGAs (ONNX in, bitstream/accelerator out) and requires its own toolchain (Python 3.10–3.11, Vivado/Vitis/Vitis HLS). See that repo’s README for install and build steps.

#### Jetson — NVIDIA driver, CUDA, TensorRT

Required to build TensorRT `.engine` artifacts (on an NVIDIA host or on the Jetson device itself).

1. Install the [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) and [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) for your GPU / Jetson.
2. Run `pip install -e .` from [§3](#3-install-python-dependencies) (includes TensorRT **10.x**). INT8 calibration also needs `calibration_data` in `platforms.json`.

Build on an NVIDIA host or on the target Jetson device — CUDA is required for TensorRT engine generation.

#### ARM — ncnn

No extra host install beyond `pip install -e .` — ncnn and pnnx are Python dependencies used for PyTorch/Keras → ncnn export.

#### MCU — STEdgeAI + STM32CubeIDE

Required for STEdgeAI compile and STM32N6 firmware build (Keras image classification and YOLO object detection).

1. Install [STEdgeAI](https://www.st.com/en/embedded-software/x-cube-ai.html) (Linux installer; default path below).
2. Install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) for firmware build.

```bash
export STM32_AI_EXE=/opt/ST/STEdgeAI/4.0/Utilities/linux/stedgeai
export STM32_CUBE_IDE_EXE=/opt/st/stm32cubeide_2.1.1/stm32cubeide
```

Adjust paths to match your install locations. Add these lines to `~/.bashrc` if you deploy to MCU often.

### 6. Verify installation

```bash
edge-ai-deploy --help
python -c "import torch, tensorflow; print('torch', torch.__version__, 'tf', tensorflow.__version__)"

# Verify target-specific tools (Docker, Vitis-AI image, STEdgeAI, STM32CubeIDE, TensorRT, CUDA)
edge-ai-deploy doctor
```

See [Host pre-flight checks](#host-pre-flight-checks) for what each check covers.

## Run

### PyTorch → FPGA (MNIST example)

```bash
edge-ai-deploy data/mnist_model.pt \
  --model-class data/model_class/mnist_net.py:Net \
  --input-shape 1,28,28 \
  --task image_classification \
  --latency-type ultra_low \
  --max-cost-usd 2000 \
  --auto-select
```

### PyTorch → ARM

```bash
edge-ai-deploy data/mnist_model.pt \
  --model-class data/model_class/mnist_net.py:Net \
  --input-shape 1,28,28 \
  --task image_classification \
  --latency-type moderate \
  --auto-select
```

### Keras → MCU (image classification)

```bash
edge-ai-deploy data/mnist_model.h5 \
  --input-shape 1,28,28 \
  --task image_classification \
  --latency-type low \
  --board-id stm32n6570_dk \
  --auto-select
```

### Ultralytics YOLO → FPGA (object detection)

Deploy a YOLO checkpoint to the ZCU104 via Vitis AI. Use `--board-id zcu104` (or `--toolchain vitis_ai`) so DPU-compatible Ultralytics layers are patched on the host before load. ZCU104 costs ~$1678 — raise `--max-cost-usd` above the default `$500`.

**Do not use stock Ultralytics weights for YOLOv8 / YOLOv11 / YOLOv26** — train with [yolov8-ultralytics](https://github.com/surajkarki66/yolov8-ultralytics) or [yolov26-ultralytics](https://github.com/surajkarki66/yolov26-ultralytics) first (see **FPGA: train with DPU-compatible forks first** above).

**YOLOv8 checkpoint** (from [yolov8-ultralytics](https://github.com/surajkarki66/yolov8-ultralytics)):

```bash
edge-ai-deploy data/best_yolov8n.pt \
  --input-shape 3,416,416 \
  --task object_detection \
  --latency-type ultra_low \
  --board-id zcu104 \
  --max-cost-usd 2000 \
  --verbose
```

**YOLOv11 / YOLOv26 checkpoint** (from [yolov26-ultralytics](https://github.com/surajkarki66/yolov26-ultralytics)):

```bash
edge-ai-deploy data/best_yolov26.pt \
  --input-shape 3,416,416 \
  --task object_detection \
  --latency-type ultra_low \
  --board-id zcu104 \
  --max-cost-usd 2000 \
  --verbose
```

### Ultralytics YOLO → MCU (object detection)

STM32N6 object detection exports YOLO to TFLite, then runs STEdgeAI compile and the bundled YOLO firmware project. Set `yolo_export.calibration_data` in `platforms.json` to an Ultralytics dataset `.yaml` (default: `data/calibration/data.yaml`).

```bash
edge-ai-deploy data/yolov8n.pt \
  --input-shape 3,640,640 \
  --task object_detection \
  --latency-type low \
  --board-id stm32n6570_dk \
  --auto-select
```

Requires `STM32_AI_EXE` and `STM32_CUBE_IDE_EXE` (see host pre-flight checks below).

`--input-shape` is **required** (`C,H,W`, e.g. `3,224,224` for RGB).

`--task` is **required**:

| Value                    | Use case                               |
| ------------------------ | -------------------------------------- |
| `image_classification` | Classifiers (e.g. MNIST, CIFAR10)      |
| `object_detection`     | Ultralytics YOLO checkpoints (`.pt`) |

`--latency-type` is **required**:

| Value         | Typical platforms                             |
| ------------- | --------------------------------------------- |
| `ultra_low` | FPGA (`zcu104`)                             |
| `low`       | MCU (`stm32n6570_dk`)                       |
| `medium`    | NVIDIA Jetson (`jetson_nano`, `agx_orin`) |
| `moderate`  | Raspberry Pi (`raspberry_pi_4`)             |

Or invoke the module directly:

```bash
python -m edge_ai_deployment_pipeline.main data/mnist_model.pt \
  --model-class data/model_class/mnist_net.py:Net \
  --input-shape 1,28,28 \
  --task image_classification \
  --latency-type ultra_low \
  --auto-select
```

### Board selection

After constraint ranking, the CLI either:

1. **Prompts** you to pick a rank (default), or
2. Uses `**--board-id zcu104`** to deploy to a specific board (also selects Vitis-AI Ultralytics DPU patching at intake for YOLO), or
3. Uses `**--auto-select`** to take rank #1 without prompting.

Example with budget filters:

```bash
edge-ai-deploy data/mnist_model.pt \
  --model-class data/model_class/mnist_net.py:Net \
  --input-shape 1,28,28 \
  --task image_classification \
  --latency-type ultra_low \
  --max-power-w 15 \
  --max-cost-usd 2000 \
  --auto-select
```

JSON output includes `rankings` (all scored boards) and `selected_platform` (the board used for deployment).

### CLI options

| Flag                     | Default                   | Description                                                                                                                               |
| ------------------------ | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `model`                | —                        | **Required.** `.pt`/`.pth` weights or `.keras`/`.h5` model                                                                  |
| `--task`               | —                        | **Required.** `image_classification` or `object_detection`                                                                      |
| `--latency-type`       | —                        | **Required.** `ultra_low`, `low`, `medium`, or `moderate`                                                                   |
| `--input-shape`        | —                        | **Required.** `C,H,W` (e.g. `3,224,224`)                                                                                        |
| `--model-class`        | —                        | **Required for `.pt`/`.pth`** except `--task object_detection` (Ultralytics YOLO). e.g. `data/model_class/mnist_net.py:Net` |
| `--model-kwargs`       | —                        | JSON constructor kwargs, e.g.`'{"num_classes": 10}'`                                                                                    |
| `-o`, `--output-dir` | `data/artifacts`        | Base dir; each run writes to`data/artifacts/<run-id>/`                                                                                  |
| `--clean`              | off                       | Remove previous artifact runs and logs before starting                                                                                    |
| `--check-host`         | off                       | Run host pre-flight checks and exit (no model deploy)                                                                                     |
| `--platform-config`    | bundled`platforms.json` | Custom platform catalogue                                                                                                                 |
| `--max-power-w`        | `25`                    | Boards above this TDP are**excluded**                                                                                               |
| `--max-cost-usd`       | `500`                   | Boards above this cost are**excluded**                                                                                              |
| `--toolchain`          | —                        | Force a toolchain (`tensorrt`, `vitis_ai`, `ncnn`, `stm32n6`, …)                                                                 |
| `--board-id`           | —                        | Deploy to this board (skip interactive pick)                                                                                              |
| `--auto-select`        | off                       | Use highest-ranked board without prompting                                                                                                |
| `-v`, `--verbose`    | off                       | Debug logging                                                                                                                             |

### Host pre-flight checks

Verify Docker, Vitis-AI, STEdgeAI, STM32CubeIDE, TensorRT, and CUDA before a long deploy run:

```bash
edge-ai-deploy doctor
# or
edge-ai-deploy --check-host
edge-ai-doctor
```

Checks include:

| Check                                        | Required for                                     |
| -------------------------------------------- | ------------------------------------------------ |
| Python (torch, tensorflow)                   | All steps                                        |
| Docker                                       | FPGA (Vitis AI)                                  |
| `xilinx/vitis-ai-pytorch-cpu:latest` image | FPGA quantize + DPU compile                      |
| `STM32_AI_EXE`                             | MCU (STEdgeAI compile)                           |
| `STM32_CUBE_IDE_EXE`                       | MCU firmware build (Keras compile / YOLO deploy) |
| TensorRT 10.x                                | Jetson INT8                                      |
| CUDA                                         | Jetson TensorRT engine build                     |

### Run output layout

Each invocation gets a unique `run_id`. Artifacts go under `data/artifacts/<run-id>/<board-id>/`. Logs are written to `logs/pipeline_<run-id>.log`. Old runs and logs are rotated automatically (50 logs, 20 artifact runs retained). Use `--clean` to wipe them before a new run.

JSON output includes `run_id` and `artifacts_dir`.

## Constraint scoring

For each board matching `--latency-type`, the best toolchain per board is scored and **all** candidates are **ranked** (highest score first). The compatibility penalty is `unsupported_layer_ratio × 100` (e.g. 10% unsupported layers → −10 score).

**Score weights:** power 50%, cost 50%.

**Memory check:** `model_size_mb + max(peak_activation_memory_mb, inference_ram_mb)` must fit in the board's `memory_mb`. If not, a **−50** penalty is applied.

Boards whose `tdp_watts` or `cost_usd` exceed `--max-power-w` or `--max-cost-usd` are **excluded**. The pipeline exits with an error listing each rejected board and why.

## Compatibility

Each toolchain uses either an **allowlist** (FPGA Vitis AI, MCU STM32N6) or **denylist** (Jetson TensorRT, ARM ncnn) of layer/operation names.

| Status               | Meaning                                                                           |
| -------------------- | --------------------------------------------------------------------------------- |
| `fully_compatible` | No unsupported operations                                                         |
| `partial`          | Some unsupported layers, but ≤ 50% of total layer count                          |
| `incompatible`     | Framework mismatch and/or > 50% unsupported layers — ranked with a score penalty |

Layer compatibility is measured by **layer count** (e.g. 1 unsupported `Einsum` out of 87 layers ≈ 1%), not by unique layer type names. Partial and incompatible boards are included in rankings; the score penalty is proportional to `unsupported_layer_ratio` (up to −100).

## Deployment notes

The **deployment** step runs the platform adapter for the selected board. Expect target-specific failures if host tools are missing.

| Target           | Output                    | Notes                                                                                                                                                                                                                                                                                                                  |
| ---------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FPGA**   | `.xmodel`               | **Vitis AI** (default in this pipeline): PyTorch PTQ + DPU compile in Docker. Ultralytics YOLO: DPU `head.py`/`block.py` patch, then quantize via `YOLO()` load. **FINN+:** use [surajkarki66/finn-plus](https://github.com/surajkarki66/finn-plus) separately — not produced by `edge-ai-deploy`. |
| **Jetson** | TensorRT`.engine`       | PyTorch → ONNX → TensorRT. Requires CUDA. INT8 needs`calibration_data` and TensorRT 10.x.                                                                                                                                                                                                                          |
| **ARM**    | ncnn`.param` / `.bin` | PyTorch → ONNX → pnnx. Keras via tf2onnx → onnx2torch.                                                                                                                                                                                                                                                              |
| **MCU**    | STEdgeAI bundle           | **IC:** Keras → TFLite → `stedgeai` compile. **OD:** Ultralytics → TFLite → STEdgeAI → STM32CubeIDE firmware build. Requires `mako`.                                                                                                                                                              |

Artifacts are written under `<output-dir>/<run-id>/<board-id>/` (default `data/artifacts/<run-id>/`).

Logs are written under `logs/pipeline_<run-id>.log`.

## Project layout

```
edge_ai_deployment_pipeline/
├── src/edge_ai_deployment_pipeline/
│   ├── main.py                 # CLI entry (edge-ai-deploy)
│   ├── core/                   # pipeline, CLI, pytorch_loader, board selection
│   ├── steps/                  # intake, profiler, compatibility, constraint, deployment
│   ├── adapters/               # jetson, fpga, mcu, arm toolchains
│   │   └── fpga/toolchains/ultralytics_dpu/  # DPU-compatible YOLO head/block
│   └── config/platforms.json   # platform catalogue
├── data/                       # centralized data root
│   ├── model_class/mnist_net.py  # example PyTorch network definition
│   ├── calibration/            # sample calibration images + data.yaml (YOLO/MCU/FPGA PTQ)
│   └── artifacts/              # deployment output (<run-id>/<board-id>/); add your own .pt checkpoints under data/
├── logs/                       # pipeline logs (pipeline_<run-id>.log)
├── .github/workflows/ci.yml    # ruff, mypy
├── pyproject.toml
├── requirements.txt
└── ruff.toml
```
