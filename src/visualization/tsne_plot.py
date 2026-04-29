import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def compute_tsne(X, random_state=42, perplexity=30):
    """
    Compute 2D t-SNE embedding for feature matrix X.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n_samples = X.shape[0]

    # t-SNE perplexity must be smaller than number of samples
    safe_perplexity = min(perplexity, max(5, (n_samples - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )

    return tsne.fit_transform(X)


def plot_tsne_by_domain(
    X_source,
    X_target,
    output_path,
    title="t-SNE by Domain",
    random_state=42,
    perplexity=30,
):
    """
    Plot source vs target features using t-SNE.
    """
    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)

    X_all = np.vstack([X_source, X_target])
    domains = np.array(["Source"] * len(X_source) + ["Target"] * len(X_target))

    X_embedded = compute_tsne(
        X_all,
        random_state=random_state,
        perplexity=perplexity,
    )

    plt.figure(figsize=(8, 6))

    for domain in ["Source", "Target"]:
        mask = domains == domain
        plt.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            label=domain,
            alpha=0.7,
            s=18,
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_tsne_by_class(
    X_source,
    y_source,
    X_target,
    y_target,
    output_path,
    title="t-SNE by Class",
    random_state=42,
    perplexity=30,
):
    """
    Plot source + target features using t-SNE, colored by class labels.
    """
    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    y_source = np.asarray(y_source)
    y_target = np.asarray(y_target)

    X_all = np.vstack([X_source, X_target])
    y_all = np.concatenate([y_source, y_target])

    X_embedded = compute_tsne(
        X_all,
        random_state=random_state,
        perplexity=perplexity,
    )

    plt.figure(figsize=(8, 6))

    classes = np.unique(y_all)

    for cls in classes:
        mask = y_all == cls
        plt.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            label=str(cls),
            alpha=0.7,
            s=18,
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()