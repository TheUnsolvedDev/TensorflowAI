import csv
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = ("dqn", "double", "dueling", "per")
COLORS = {
    "dqn": "tab:blue",
    "double": "tab:orange",
    "dueling": "tab:green",
    "per": "tab:red",
}
MAX_LOSS_POINTS = 2_000


def read_metrics(path):
    loss_steps, losses, reward_steps, rewards = [], [], [], []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["loss"]:
                loss_steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
            if row["average_reward"]:
                reward_steps.append(int(row["step"]))
                rewards.append(float(row["average_reward"]))
    return loss_steps, losses, reward_steps, rewards


def thin(steps, values):
    stride = max(1, len(steps) // MAX_LOSS_POINTS)
    return steps[::stride], values[::stride]


def main():
    figure, (loss_axis, reward_axis) = plt.subplots(2, 1, figsize=(12, 9))
    found = False

    for model in MODELS:
        for metrics_path in sorted((Path("runs") / model).glob("*/metrics.csv")):
            loss_steps, losses, reward_steps, rewards = read_metrics(metrics_path)
            label = f"{model} {metrics_path.parent.name}"
            color = COLORS[model]
            if losses:
                plot_steps, plot_losses = thin(loss_steps, losses)
                loss_axis.plot(plot_steps, plot_losses, color=color, alpha=0.7, label=label)
                found = True
            if rewards:
                reward_axis.plot(
                    reward_steps,
                    rewards,
                    color=color,
                    marker="o",
                    alpha=0.8,
                    label=label,
                )
                found = True

    if not found:
        raise SystemExit("No timestamped metrics found under runs/<model>/<timestamp>/")

    loss_axis.set(title="Training loss", xlabel="Transitions", ylabel="Huber loss")
    reward_axis.set(
        title="Evaluation reward",
        xlabel="Transitions",
        ylabel="Average reward (100 episodes)",
    )
    for axis in (loss_axis, reward_axis):
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=7)

    figure.tight_layout()
    output = Path("runs/all_models.png")
    figure.savefig(output, dpi=160)
    print(f"saved {output}")
    plt.show()


if __name__ == "__main__":
    main()
