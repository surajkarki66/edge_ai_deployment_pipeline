# Edge AI Deployment Pipeline (Computer Vision)

Modular pipeline for **computer-vision models** — profile, check edge compatibility, score constraints, and emit a deployable artifact.

## Layout

```
edge_ai_deployment_pipeline/
├── src/                 # application source
│   ├── core/
│   ├── stages/
│   ├── adapters/
│   ├── config/          # platforms.json
│   └── operators/
├── main.py              # CLI entry point
└── pyproject.toml
```

## Run

From the repository root:

```bash
pip install -r requirements.txt
python main.py path/to/model.pt --max-latency-ms 20 --max-power-w 15 --max-cost-usd 200
```

Optional: `-o artifacts/`, `--platform-config path/to/platforms.json`, `--toolchain tensorrt`, `-v`.

Default platform catalogue: `src/config/platforms.json`.
