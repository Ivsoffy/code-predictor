# code-predictor

Главный inference-файл: `pipeline.py`.

Пример применения находится в `example.ipynb`.

## Setup

```powershell
uv sync
```

## Training

```powershell
uv run python train.py --config mlflow_config/train_config.yaml
```

Локальные данные, веса моделей и MLflow runs не коммитятся. Для пустых директорий в репозитории оставлены `data/.gitkeep` и `models/.gitkeep`.
