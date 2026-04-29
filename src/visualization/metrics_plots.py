import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(metrics_path: str):
    """
    Load metrics from a JSON file.

    Parameters:
        metrics_path (str): path to metrics JSON

    Returns:
        dict: loaded metrics
    """
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_branch1_summary(metrics_path: str, output_path: str):
    """
    Create a simple summary plot for Branch1 results.

    Parameters:
        metrics_path (str): path to saved metrics JSON
        output_path (str): path to save the output figure
    """
    metrics = load_metrics(metrics_path)

    labels = [
        "MMD Before",
        "MMD After",
        "Accuracy Before",
        "Accuracy After",
    ]

    values = [
        metrics["mmd_before_adaptation"],
        metrics["mmd_after_adaptation"],
        metrics["baseline_accuracy_on_target"],
        metrics["adapted_accuracy_on_target"],
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title("Branch1 Summary Metrics")
    plt.ylabel("Value")
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()