import argparse
import json
import re
from pathlib import Path
from typing import Any

import evaluate as hf_evaluate
import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

TRAINING_COMPLETED_MARKER = "training_completed.json"
MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "nil", "undefined"}


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_codes(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        codes = json.load(f)
    print(f"Loaded {len(codes)} classification codes")
    return codes


def read_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return ""
    return re.sub(r"\s+", " ", text.lower())


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return ""
    return text


def make_input_text(row: pd.Series, columns: dict[str, Any]) -> str:
    parts = []
    industry = clean_text(row.get(columns["industry"]))
    if industry:
        parts.append(industry)

    for dep_col in columns["departments"]:
        dep = clean_text(row.get(dep_col))
        if dep:
            parts.append(dep)

    job_title = clean_text(row.get(columns["job_title"]))
    if job_title:
        parts.append(job_title)

    if not parts:
        return ""
    return ". ".join(parts) + "."


def prepare_dataset(
    df: pd.DataFrame,
    codes: dict[str, Any],
    columns: dict[str, Any],
) -> pd.DataFrame:
    df = df.copy()
    code_col = columns["code"]
    df[code_col] = df[code_col].astype(str)

    valid_codes = set(codes.keys())
    print("Размер датасета: ", df.shape[0])
    missing_codes = sorted(set(df.loc[~df[code_col].isin(valid_codes), code_col]))
    df = df.loc[df[code_col].isin(valid_codes)].copy()

    print("Размер датасета после удаления несуществующих кодов: ", df.shape[0])
    print(f"Несуществующие коды: {missing_codes}")

    df["input_text"] = df.apply(lambda row: make_input_text(row, columns), axis=1)
    df["target_text"] = df[code_col].apply(
        lambda code: str(codes[code]["description"])
    )

    df = df[["input_text", "target_text"]].dropna()
    df["input_text"] = df["input_text"].astype(str)
    df["target_text"] = df["target_text"].astype(str)
    df = df.loc[df["input_text"].str.len() > 0].copy()
    return df


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: T5TokenizerFast,
    input_max_length: int,
    target_max_length: int,
) -> Dataset:
    def preprocess(batch):
        inputs = ["summarize: " + text for text in batch["input_text"]]
        model_inputs = tokenizer(
            inputs,
            max_length=input_max_length,
            truncation=True,
            padding="max_length",
        )

        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=target_max_length,
            truncation=True,
            padding="max_length",
        )
        pad_token_id = tokenizer.pad_token_id
        model_inputs["labels"] = [
            [token if token != pad_token_id else -100 for token in label]
            for label in labels["input_ids"]
        ]
        return model_inputs

    return dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
    )


def decode_predictions(
    tokenizer: T5TokenizerFast,
    predictions: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[list[str], list[str]]:
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return (
        [normalize_text(text) for text in pred_texts],
        [normalize_text(text) for text in label_texts],
    )


def build_compute_metrics(tokenizer: T5TokenizerFast, f1_average: str):
    rouge = hf_evaluate.load("rouge")

    def compute_metrics(eval_pred):
        pred_texts, label_texts = decode_predictions(
            tokenizer,
            eval_pred.predictions,
            eval_pred.label_ids,
        )
        rouge_metrics = rouge.compute(predictions=pred_texts, references=label_texts)
        exact_match = np.mean(
            [pred == label for pred, label in zip(pred_texts, label_texts)]
        )

        return {
            "f1": float(
                f1_score(
                    label_texts,
                    pred_texts,
                    average=f1_average,
                    zero_division=0,
                )
            ),
            "exact_match": float(exact_match),
            "rougeL": float(rouge_metrics["rougeL"]),
            "rouge1": float(rouge_metrics["rouge1"]),
            "rouge2": float(rouge_metrics["rouge2"]),
        }

    return compute_metrics


def output_dir_index(path: Path, base_path: Path) -> int | None:
    if path.name == base_path.name:
        return 0

    prefix = f"{base_path.name}_"
    if path.name.startswith(prefix):
        suffix = path.name[len(prefix) :]
        if suffix.isdigit():
            return int(suffix)
    return None


def existing_output_dirs(base_path: Path) -> list[Path]:
    parent = base_path.parent
    if not parent.exists():
        return []

    output_dirs = []
    for path in parent.iterdir():
        if not path.is_dir():
            continue
        idx = output_dir_index(path, base_path)
        if idx is not None:
            output_dirs.append((idx, path))

    output_dirs.sort(key=lambda item: item[0])
    return [path for _, path in output_dirs]


def is_training_completed(output_dir: Path) -> bool:
    if (output_dir / TRAINING_COMPLETED_MARKER).exists():
        return True

    has_config = (output_dir / "config.json").exists()
    has_weights = any(
        (output_dir / filename).exists()
        for filename in (
            "model.safetensors",
            "pytorch_model.bin",
            "tf_model.h5",
            "flax_model.msgpack",
        )
    )
    return has_config and has_weights


def create_next_output_dir(base_path: Path, output_dirs: list[Path]) -> Path:
    existing_indexes = [
        idx
        for path in output_dirs
        if (idx := output_dir_index(path, base_path)) is not None
    ]
    next_idx = max(existing_indexes, default=-1) + 1

    while True:
        candidate = (
            base_path
            if next_idx == 0
            else base_path.with_name(f"{base_path.name}_{next_idx}")
        )
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        next_idx += 1


def prepare_output_dir(base_dir: str | Path) -> tuple[str, str | None]:
    base_path = Path(base_dir)
    output_dirs = existing_output_dirs(base_path)

    if not output_dirs:
        base_path.mkdir(parents=True, exist_ok=False)
        return str(base_path), None

    latest_dir = output_dirs[-1]
    if not is_training_completed(latest_dir):
        last_checkpoint = get_last_checkpoint(str(latest_dir))
        if last_checkpoint:
            print("Resuming training from checkpoint: ", last_checkpoint)
        else:
            print(
                "Unfinished training dir found without checkpoints. "
                "Starting it from scratch: ",
                latest_dir,
            )
        return str(latest_dir), last_checkpoint

    return str(create_next_output_dir(base_path, output_dirs)), None


def build_training_args(config: dict[str, Any], output_dir: str):
    training = dict(config["training"])
    if training.get("fp16") == "auto":
        training["fp16"] = torch.cuda.is_available()

    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        report_to=[],
        **training,
    )


def flatten_config(
    value: Any,
    prefix: str = "",
    result: dict[str, str] | None = None,
) -> dict[str, str]:
    if result is None:
        result = {}

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_config(child, child_prefix, result)
    elif isinstance(value, list):
        result[prefix] = ",".join(str(item) for item in value)
    else:
        result[prefix] = str(value)

    return result


def log_config_params(config: dict[str, Any]) -> None:
    params = flatten_config(config)
    for key, value in params.items():
        mlflow.log_param(key, value[:500])


def write_training_completed_marker(
    output_dir: str | Path,
    trainer: Seq2SeqTrainer,
    metrics: dict[str, float],
) -> Path:
    state = trainer.state
    marker = {
        "completed": True,
        "global_step": state.global_step,
        "epoch": state.epoch,
        "best_metric": state.best_metric,
        "best_model_checkpoint": state.best_model_checkpoint,
        "metrics": metrics,
    }

    marker_path = Path(output_dir) / TRAINING_COMPLETED_MARKER
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(marker, f, ensure_ascii=False, indent=2)
    return marker_path


def evaluate(trainer: Seq2SeqTrainer, eval_dataset: Dataset) -> dict[str, float]:
    prediction_output = trainer.predict(eval_dataset)
    metrics = {}
    for key, value in prediction_output.metrics.items():
        clean_key = key.removeprefix("test_")
        if isinstance(value, int | float | np.floating):
            metrics[clean_key] = float(value)
    return metrics


def train(config: dict[str, Any]) -> dict[str, float]:
    set_seed(int(config.get("seed", 42)))
    codes = load_codes(config["paths"]["codes"])
    columns = config["data"]["train_columns"]

    train_df = prepare_dataset(
        read_dataframe(config["paths"]["train_data"]),
        codes,
        columns,
    )

    eval_path = config["paths"].get("eval_data")
    if eval_path:
        eval_df = prepare_dataset(read_dataframe(eval_path), codes, columns)
        train_ds = Dataset.from_pandas(train_df, preserve_index=False)
        eval_ds = Dataset.from_pandas(eval_df, preserve_index=False)
    else:
        dataset = Dataset.from_pandas(train_df, preserve_index=False)
        splits = dataset.train_test_split(
            test_size=float(config["data"].get("eval_size", 0.1)),
            seed=int(config.get("seed", 42)),
        )
        train_ds = splits["train"]
        eval_ds = splits["test"]

    model_name = config["model"]["pretrained_name"]
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    train_tok = tokenize_dataset(
        train_ds,
        tokenizer,
        int(config["data"]["input_max_length"]),
        int(config["data"]["target_max_length"]),
    )
    eval_tok = tokenize_dataset(
        eval_ds,
        tokenizer,
        int(config["data"]["input_max_length"]),
        int(config["data"]["target_max_length"]),
    )

    output_dir, resume_from_checkpoint = prepare_output_dir(
        config["paths"]["output_base_dir"]
    )
    print("Output dir will be: ", output_dir)

    training_args = build_training_args(config, output_dir)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    callbacks = []
    early_stopping = config.get("early_stopping", {})
    if early_stopping.get("enabled", False):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(early_stopping.get("patience", 3))
            )
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(
            tokenizer,
            config.get("evaluation", {}).get("f1_average", "macro"),
        ),
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = evaluate(trainer, eval_tok)
    marker_path = write_training_completed_marker(output_dir, trainer, metrics)

    mlflow.log_metrics(metrics)
    if "f1" in metrics:
        mlflow.log_metric("final_f1", metrics["f1"])
    mlflow.log_artifact(str(marker_path), artifact_path="training")

    if config.get("mlflow", {}).get("log_model_artifact", False):
        mlflow.log_artifacts(output_dir, artifact_path="model")

    print("Модель успешно обучена и сохранена!")
    print(f"Final F1: {metrics.get('f1')}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="mlflow_config/train_config.yaml",
        help="Path to training YAML config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_config.get("experiment_name", "code-predictor"))

    with mlflow.start_run(run_name=mlflow_config.get("run_name")):
        log_config_params(config)
        mlflow.log_artifact(str(config_path), artifact_path="config")
        train(config)


if __name__ == "__main__":
    main()
