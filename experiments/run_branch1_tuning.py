from pathlib import Path
import pandas as pd

from src.utils.io import load_digits_domain_shift
from src.branch1_vector.preprocessing import flatten_images
from src.branch1_vector.pca_pipeline import fit_pca, transform_with_pca
from src.adaptation.coral import coral
from src.adaptation.mmd import compute_mmd
from src.branch1_vector.classifier import train_knn_classifier, evaluate_classifier


RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_branch1_once(n_components=40, k=1, gamma=0.5):
    """
    Run a single Branch1 experiment and return metrics as a dict.
    Pipeline:
    image -> flatten -> PCA -> CORAL -> KNN -> accuracy
    """
    Xs_train, Xs_test, Xt_train, Xt_test, ys_train, ys_test, yt_train, yt_test = load_digits_domain_shift()

    Xs_train_flat = flatten_images(Xs_train)
    Xt_train_flat = flatten_images(Xt_train)
    Xt_test_flat = flatten_images(Xt_test)

    pca = fit_pca(Xs_train_flat, n_components=n_components)

    Xs_train_pca = transform_with_pca(pca, Xs_train_flat)
    Xt_train_pca = transform_with_pca(pca, Xt_train_flat)
    Xt_test_pca = transform_with_pca(pca, Xt_test_flat)

    mmd_before = compute_mmd(Xs_train_pca, Xt_train_pca, gamma=gamma)

    Xs_train_adapted = coral(Xs_train_pca, Xt_train_pca)

    mmd_after = compute_mmd(Xs_train_adapted, Xt_train_pca, gamma=gamma)

    baseline_model = train_knn_classifier(Xs_train_pca, ys_train, n_neighbors=k)
    baseline_acc = evaluate_classifier(baseline_model, Xt_test_pca, yt_test)

    adapted_model = train_knn_classifier(Xs_train_adapted, ys_train, n_neighbors=k)
    adapted_acc = evaluate_classifier(adapted_model, Xt_test_pca, yt_test)

    return {
        "n_components": n_components,
        "k": k,
        "gamma": gamma,
        "mmd_before": float(mmd_before),
        "mmd_after": float(mmd_after),
        "baseline_accuracy": float(baseline_acc),
        "adapted_accuracy": float(adapted_acc),
    }


def run_pca_tuning(pca_values, fixed_k=1, gamma=0.5):
    rows = []

    for n_components in pca_values:
        print("=" * 50)
        print(f"Running PCA tuning for n_components={n_components}, k={fixed_k}")

        metrics = run_branch1_once(
            n_components=n_components,
            k=fixed_k,
            gamma=gamma,
        )

        rows.append(metrics)

    df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "pca_tuning_results.csv"
    df.to_csv(output_path, index=False)

    print("\nPCA tuning results saved to:", output_path)
    print(df)

    return df


def run_knn_tuning(knn_values, fixed_n_components=40, gamma=0.5):
    rows = []

    for k in knn_values:
        print("=" * 50)
        print(f"Running KNN tuning for n_components={fixed_n_components}, k={k}")

        metrics = run_branch1_once(
            n_components=fixed_n_components,
            k=k,
            gamma=gamma,
        )

        rows.append(metrics)

    df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "knn_tuning_results.csv"
    df.to_csv(output_path, index=False)

    print("\nKNN tuning results saved to:", output_path)
    print(df)

    return df


if __name__ == "__main__":
    pca_values = [10, 20, 30, 40, 50]
    knn_values = [1, 3, 5, 7, 9]

    print("\nStarting PCA tuning...")
    run_pca_tuning(pca_values=pca_values, fixed_k=1, gamma=0.5)

    print("\nStarting KNN tuning...")
    run_knn_tuning(knn_values=knn_values, fixed_n_components=40, gamma=0.5)