import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, output_path: str, title="Confusion Matrix"):
    """
    Plot and save confusion matrix with annotations.

    Parameters:
        y_true: true labels
        y_pred: predicted labels
        output_path (str): where to save the figure
        title (str): plot title
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # TP/TN/FP/FN açıklaması (binary için anlamlı ama biz yine not düşüyoruz)
    plt.figtext(
        0.5, -0.1,
        "Diagonal = Correct predictions (TP + TN)\nOff-diagonal = Errors (FP + FN)",
        ha="center",
        fontsize=9
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()