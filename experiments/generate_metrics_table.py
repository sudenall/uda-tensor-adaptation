import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_final_metrics_table():
    rows = [
        ["Branch1", "PCA + CORAL (vector)", 10, 1, 0.1727, 0.0398, 0.5407, 0.3852, np.nan, np.nan, np.nan, np.nan, "Too aggressive compression"],
        ["Branch1", "PCA + CORAL (vector)", 20, 1, 0.0914, 0.0265, 0.6185, 0.5111, np.nan, np.nan, np.nan, np.nan, "Reduced information loss"],
        ["Branch1", "PCA + CORAL (vector)", 30, 1, 0.0630, 0.0211, 0.6241, 0.5704, np.nan, np.nan, np.nan, np.nan, "Good trade-off"],
        ["Branch1", "PCA + CORAL (vector)", 40, 1, 0.0529, 0.0189, 0.6352, 0.5704, 0.7294, 0.5950, 0.6352, 0.5703, "Best Branch1 baseline"],
        ["Branch1", "PCA + CORAL (vector)", 50, 1, 0.0471, 0.0179, 0.6278, 0.5667, np.nan, np.nan, np.nan, np.nan, "Slight overfitting"],

        ["Branch1", "KNN (vector, PCA=40)", 40, 1, 0.0529, 0.0189, 0.6352, 0.5704, np.nan, np.nan, np.nan, np.nan, "Best k"],
        ["Branch1", "KNN (vector, PCA=40)", 40, 3, 0.0529, 0.0189, 0.5722, 0.5444, np.nan, np.nan, np.nan, np.nan, "Medium k"],
        ["Branch1", "KNN (vector, PCA=40)", 40, 9, 0.0529, 0.0189, 0.5648, 0.5222, np.nan, np.nan, np.nan, np.nan, "Over-smoothing"],

        ["Branch1", "Logistic Regression (vector)", 40, np.nan, np.nan, np.nan, 0.4019, 0.4778, np.nan, np.nan, np.nan, np.nan, "Linear model benefits from adaptation"],
        ["Branch1", "SVM (vector)", 40, np.nan, np.nan, np.nan, 0.5519, 0.5500, np.nan, np.nan, np.nan, np.nan, "Stable, limited gain"],

        ["Branch2", "2D-PCA (tensor baseline)", 4, 3, 0.0177, np.nan, 0.5833, np.nan, np.nan, np.nan, np.nan, np.nan, "Tensor-aware baseline"],
        ["Branch2", "2D-PCA + CORAL (tensor)", 4, 3, 0.0177, 0.0124, 0.5833, 0.5796, np.nan, np.nan, np.nan, np.nan, "Strong alignment but degraded discriminative structure"],
        ["Branch2", "UDA-TFL-inspired (tensor)", 4, 3, 0.0177, 0.0146, 0.5833, 0.5944, np.nan, np.nan, np.nan, np.nan, "BEST: optimal balance between domain alignment and class separability"],
    ]

    columns = [
        "Branch",
        "Method",
        "n_components",
        "k",
        "MMD Before",
        "MMD After",
        "Acc Before",
        "Acc After",
        "Precision Before",
        "Precision After",
        "Recall Before",
        "Recall After",
        "Note",
    ]

    df = pd.DataFrame(rows, columns=columns)
    df["Δ Acc (After - Before)"] = df["Acc After"] - df["Acc Before"]

    ordered_cols = [
        "Branch",
        "Method",
        "n_components",
        "k",
        "MMD Before",
        "MMD After",
        "Acc Before",
        "Acc After",
        "Δ Acc (After - Before)",
        "Precision Before",
        "Precision After",
        "Recall Before",
        "Recall After",
        "Note",
    ]

    return df[ordered_cols]


def save_metrics_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def format_value(value, column):
    if pd.isna(value):
        return "-"

    if column in ["n_components", "k"]:
        return str(int(value))

    if isinstance(value, (float, int, np.floating, np.integer)):
        return f"{value:.4f}"

    return str(value)


def save_metrics_image(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    display_df = df.copy()

    for col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: format_value(x, col))

    fig_height = max(8, len(display_df) * 0.48)
    fig_width = 28

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)

    best_method = "UDA-TFL-inspired (tensor)"

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(0.08)

        if row > 0:
            method_text = display_df.iloc[row - 1]["Method"]

            if method_text == best_method:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#DFF2BF")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = build_final_metrics_table()

    csv_path = "results/final_metrics.csv"
    image_path = "results/figures/final_metrics_table.png"

    save_metrics_csv(df, csv_path)
    save_metrics_image(df, image_path)

    print(f"Final metrics CSV saved to: {csv_path}")
    print(f"Final metrics table image saved to: {image_path}")


if __name__ == "__main__":
    main()