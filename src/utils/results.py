import json
from pathlib import Path


def save_metrics(metrics: dict, output_path: str):
    """
    Save experiment metrics as a JSON file.

    Parameters:
        metrics (dict): dictionary containing experiment results
        output_path (str): path to output JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)