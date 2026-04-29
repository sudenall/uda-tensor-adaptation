import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_da_table():
    rows = [
        [
            "Baseline",
            "2D-PCA",
            "Tensor",
            "None",
            0.0177,
            np.nan,
            0.5833,
            np.nan,
            np.nan,
            "No adaptation applied",
        ],
        [
            "Vector DA",
            "PCA + CORAL",
            "Vector",
            "CORAL",
            0.0529,
            0.0189,
            0.6352,
            0.5704,
            -0.0648,
            "Alignment improves, but classification performance drops",
        ],
        [
            "Tensor DA",
            "2D-PCA + CORAL",
            "Tensor",
            "CORAL",
            0.0177,
            0.0124,
            0.5833,
            0.5796,
            -0.0037,
            "Strong alignment, but discriminative structure is weakened",
        ],
        [
            "Tensor DA",
            "UDA-TFL-inspired",
            "Tensor",
            "MMD + Class-aware",
            0.0177,
            0.0146,
            0.5833,
            0.5944,
            0.0111,
            "BEST: optimal balance between domain alignment and class separability",
        ],
    ]

    columns = [
        "Group",
        "Method",
        "Representation",
        "DA Method",
        "MMD Before",
        "MMD After",
        "Acc Before",
        "Acc After",
        "Δ Acc",
        "Key Finding",
    ]

    return pd.DataFrame(rows, columns=columns)


def format_value(value, column):
    if pd.isna(value):
        return "N/A"

    if isinstance(value, (float, int, np.floating, np.integer)):
        return f"{value:.4f}"

    if column == "Key Finding":
        return "\n".join(textwrap.wrap(str(value), width=34))

    return str(value)


def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def save_table_image(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    display_df = df.copy()

    for col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: format_value(x, col))

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    col_widths = [
        0.08,  # Group
        0.12,  # Method
        0.10,  # Representation
        0.12,  # DA Method
        0.09,  # MMD Before
        0.09,  # MMD After
        0.09,  # Acc Before
        0.09,  # Acc After
        0.08,  # Δ Acc
        0.22,  # Key Finding
    ]

    for col_idx, width in enumerate(col_widths):
        for row_idx in range(len(display_df) + 1):
            table[(row_idx, col_idx)].set_width(width)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(0.14)

        if row > 0:
            method = display_df.iloc[row - 1]["Method"]

            if method == "UDA-TFL-inspired":
                cell.set_facecolor("#DFF2BF")
                cell.set_text_props(weight="bold")

    plt.figtext(
        0.5,
        0.03,
        "Note: Baseline methods do not perform domain adaptation; therefore 'After' metrics are not applicable.\n"
        "UDA-TFL-inspired optimizes both domain alignment and class separability, unlike CORAL which only aligns distributions.",
        wrap=True,
        horizontalalignment="center",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = build_da_table()

    csv_path = "results/final_da_table.csv"
    image_path = "results/figures/final_da_table.png"

    save_csv(df, csv_path)
    save_table_image(df, image_path)

    print(f"Domain Adaptation CSV saved to: {csv_path}")
    print(f"Domain Adaptation table image saved to: {image_path}")


if __name__ == "__main__":
    main()