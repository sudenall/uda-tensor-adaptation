import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def compute_tsne(X, random_state=42, perplexity=30):
    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]
    perplexity = min(perplexity, max(5, (n_samples - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X)


def plot_comparison(
    Xs_list,
    Xt_list,
    ys,
    yt,
    titles,
    output_path="results/figures/final_tsne_comparison.png",
):
    """
    Xs_list: [Xs_pca, Xs_coral, Xs_uda]
    Xt_list: [Xt_pca, Xt_coral, Xt_uda]
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i in range(3):
        Xs = Xs_list[i]
        Xt = Xt_list[i]

        X_all = np.vstack([Xs, Xt])
        domains = np.array(["Source"] * len(Xs) + ["Target"] * len(Xt))
        labels = np.concatenate([ys, yt])

        X_emb = compute_tsne(X_all)

        # --- DOMAIN ROW ---
        ax = axes[0, i]
        for d in ["Source", "Target"]:
            mask = domains == d
            ax.scatter(
                X_emb[mask, 0],
                X_emb[mask, 1],
                label=d,
                s=12,
                alpha=0.7,
            )
        ax.set_title(titles[i] + " (Domain)")

        # --- CLASS ROW ---
        ax = axes[1, i]
        classes = np.unique(labels)
        for c in classes:
            mask = labels == c
            ax.scatter(
                X_emb[mask, 0],
                X_emb[mask, 1],
                s=12,
                alpha=0.7,
            )
        ax.set_title(titles[i] + " (Class)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()