import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_model(log_dir):
    steps_loss, losses, steps_reward, rewards = [], [], [], []
    with (log_dir / "metrics.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            if row["loss"]:
                steps_loss.append(int(row["step"]))
                losses.append(float(row["loss"]))
            if row["average_reward"]:
                steps_reward.append(int(row["step"]))
                rewards.append(float(row["average_reward"]))

    if losses:
        plt.figure()
        plt.plot(steps_loss, losses)
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title(f"{log_dir.name} loss")
        plt.tight_layout()
        plt.savefig(log_dir / "loss.png")
        plt.close()
    if rewards:
        plt.figure()
        plt.plot(steps_reward, rewards, marker="o")
        plt.xlabel("Training step")
        plt.ylabel("Average reward")
        plt.title(f"{log_dir.name} average reward")
        plt.tight_layout()
        plt.savefig(log_dir / "average_reward.png")
        plt.close()


for metrics_file in sorted(Path("runs").rglob("metrics.csv")):
    plot_model(metrics_file.parent)
