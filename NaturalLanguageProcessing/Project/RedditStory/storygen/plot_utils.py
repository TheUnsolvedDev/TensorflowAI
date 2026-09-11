"""Small plotting and model-report helpers."""

from __future__ import annotations

import json
from pathlib import Path


def write_model_summary(model, output_path: str | Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append, expand_nested=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_history_json(history: dict, output_path: str | Path) -> None:
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)


def plot_model_graph(model, output_path: str | Path) -> bool:
    try:
        from tensorflow.keras.utils import plot_model

        plot_model(model, to_file=str(output_path), show_shapes=True, expand_nested=True, dpi=144)
        return True
    except Exception:
        return False


def plot_training_log(log_path: str | Path, output_path: str | Path) -> bool:
    log_path = Path(log_path)
    if not log_path.exists():
        return False

    train_steps = []
    train_loss = []
    train_acc = []
    val_steps = []
    val_loss = []
    val_acc = []
    val_ppl = []

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "train_loss" in row:
                train_steps.append(row["step"])
                train_loss.append(row["train_loss"])
                train_acc.append(row.get("train_token_accuracy", 0.0))
            elif "loss" in row:
                val_steps.append(row["step"])
                val_loss.append(row["loss"])
                val_acc.append(row.get("token_accuracy", 0.0))
                val_ppl.append(row.get("perplexity", 0.0))

    if not train_steps and not val_steps:
        return False

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    if train_steps:
        axes[0].plot(train_steps, train_loss, label="train_loss", color="#0f766e")
        axes[1].plot(train_steps, train_acc, label="train_acc", color="#1d4ed8")
    if val_steps:
        axes[0].plot(val_steps, val_loss, label="val_loss", color="#b91c1c")
        axes[1].plot(val_steps, val_acc, label="val_acc", color="#7c3aed")
        axes[2].plot(val_steps, val_ppl, label="val_perplexity", color="#c2410c")

    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy")
    axes[2].set_ylabel("Perplexity")
    axes[2].set_xlabel("Step")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=144)
    plt.close(fig)
    return True
