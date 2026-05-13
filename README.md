# code-predictor

The main inference entry point is `pipeline.py`.

`example.ipynb` shows a basic usage example.

## Setup

```powershell
uv sync
```

## Training

```powershell
uv run train.py --config mlflow_config/train_config.yaml
```

The training configuration is stored in `mlflow_config/train_config.yaml`.
Training parameters and final metrics, including `f1`, are logged to MLflow.

Local datasets, model weights, MLflow runs, and generated artifacts are ignored by git.
The repository keeps empty `data/` and `models/` directories via `.gitkeep` files.
