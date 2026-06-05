# Edge AI Deployment Pipeline (Computer Vision)

Modular pipeline for **computer-vision models** — profile, check edge compatibility, and rank boards against your constraints.

## Layout

```
edge_ai_deployment_pipeline/          # project root
├── src/edge_ai_deployment_pipeline/
│   ├── main.py                       # CLI entry point
│   ├── core/
│   ├── stages/
│   ├── adapters/
│   └── config/                       # platforms.json
├── setup.py
├── pyproject.toml
├── requirements.txt
└── ruff.toml
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or install from `requirements.txt` first, then the package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

Optional developer tools:

```bash
pip install -e ".[dev]"
```

## Run

After install, use the CLI entry point:

```bash
edge-ai-deploy path/to/model.pt --input-shape 3,224,224 --max-latency-ms 20 --max-power-w 15 --max-cost-usd 200 --toolchain tensorrt
```

Or invoke the package module directly:

```bash
python -m edge_ai_deployment_pipeline.main path/to/model.pt --input-shape 3,224,224
```

Optional: `--platform-config path/to/platforms.json`, `--max-latency-ms`, `--max-power-w`, `--max-cost-usd`, `--toolchain`, `-v`.

Default platform catalogue: bundled `config/platforms.json` (inside the installed package).
