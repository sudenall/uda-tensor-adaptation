from src.utils.io import load_digits_domain_shift
from src.branch2_tensor.tensor_preprocessing import prepare_tensor_images
from src.branch2_tensor.twodpca import (
    fit_transform_2dpca,
    transform_2dpca,
    flatten_projected_features,
)
from src.branch2_tensor.classifier import train_and_evaluate_knn
from src.adaptation.coral import coral
from src.adaptation.mmd import compute_mmd
from src.visualization.tsne_plot import (
    plot_tsne_by_domain,
    plot_tsne_by_class,
)
import os
from src.branch2_tensor.uda_tfl_inspired import (
    fit_uda_tfl_inspired,
    transform_uda_tfl_inspired,
)
from src.visualization.tsne_comparison import plot_comparison



def run_branch2_experiment(n_components=4, n_neighbors=3):
    """
    Run Branch2 baseline + CORAL adaptation experiment.

    Correct methodology:
    - Fit 2D-PCA on source train
    - Transform source train, target train, target test
    - Fit/apply CORAL using target train, not target test
    - Evaluate only on target test
    """

    # 1. Load source/target domain data
    Xs_train, Xs_test, Xt_train, Xt_test, ys_train, ys_test, yt_train, yt_test = (
        load_digits_domain_shift()
    )

    # 2. Prepare tensor images
    Xs_train = prepare_tensor_images(Xs_train, name="Xs_train")
    Xt_train = prepare_tensor_images(Xt_train, name="Xt_train")
    Xt_test = prepare_tensor_images(Xt_test, name="Xt_test")

    # 3. Fit 2D-PCA on source train
    Xs_train_proj, W, mean_image = fit_transform_2dpca(
        Xs_train,
        n_components=n_components,
    )

    # 4. Transform target train and target test using the same projection
    Xt_train_proj = transform_2dpca(
        Xt_train,
        W=W,
        mean_image=mean_image,
    )

    Xt_test_proj = transform_2dpca(
        Xt_test,
        W=W,
        mean_image=mean_image,
    )

    # 5. Convert projected tensor features into compact vectors
    Xs_train_flat = flatten_projected_features(Xs_train_proj)
    Xt_train_flat = flatten_projected_features(Xt_train_proj)
    Xt_test_flat = flatten_projected_features(Xt_test_proj)

    # 6. CORAL adaptation
    # IMPORTANT:
    # CORAL uses target train / unlabeled target adaptation set,
    # not target test.
    Xs_train_adapted = coral(Xs_train_flat, Xt_train_flat)

    # 7. Measure discrepancy
    # before/after measured against target train because adaptation uses target train
    mmd_before_train = compute_mmd(Xs_train_flat, Xt_train_flat)
    mmd_after_train = compute_mmd(Xs_train_adapted, Xt_train_flat)

    # optional: check how adapted source relates to target test distribution
    # this is diagnostic only, not used for fitting
    mmd_before_test = compute_mmd(Xs_train_flat, Xt_test_flat)
    mmd_after_test = compute_mmd(Xs_train_adapted, Xt_test_flat)

    # 8. Evaluate baseline and adapted classifiers on target test only
    baseline_acc = train_and_evaluate_knn(
        Xs_train_flat,
        ys_train,
        Xt_test_flat,
        yt_test,
        n_neighbors=n_neighbors,
    )

    adapted_acc = train_and_evaluate_knn(
        Xs_train_adapted,
        ys_train,
        Xt_test_flat,
        yt_test,
        n_neighbors=n_neighbors,
    )

    # 9. Print results
    print("=== Branch2 Experiment: 2D-PCA + CORAL ===")
    print(f"2D-PCA n_components: {n_components}")
    print(f"KNN n_neighbors     : {n_neighbors}")
    print()
    print(f"Source train shape        : {Xs_train.shape}")
    print(f"Target train shape        : {Xt_train.shape}")
    print(f"Target test shape         : {Xt_test.shape}")
    print(f"Source projected shape    : {Xs_train_proj.shape}")
    print(f"Target train proj shape   : {Xt_train_proj.shape}")
    print(f"Target test proj shape    : {Xt_test_proj.shape}")
    print(f"Source compact shape      : {Xs_train_flat.shape}")
    print(f"Target train compact shape: {Xt_train_flat.shape}")
    print(f"Target test compact shape : {Xt_test_flat.shape}")
    print(f"Adapted source shape      : {Xs_train_adapted.shape}")
    print()
    print("Discrepancy measured against target train:")
    print(f"MMD before adaptation     : {mmd_before_train}")
    print(f"MMD after adaptation      : {mmd_after_train}")
    print()
    print("Diagnostic discrepancy against target test:")
    print(f"MMD before test diagnostic: {mmd_before_test}")
    print(f"MMD after test diagnostic : {mmd_after_test}")
    print()
    print(f"Baseline accuracy on Xt_test: {baseline_acc}")
    print(f"Adapted accuracy on Xt_test : {adapted_acc}")

        # 10. Create results directory
    os.makedirs("results/figures", exist_ok=True)

    # === BEFORE ADAPTATION ===
    plot_tsne_by_domain(
        Xs_train_flat,
        Xt_test_flat,
        output_path="results/figures/branch2_tsne_domain_before.png",
        title="Branch2 t-SNE (Domain) - Before Adaptation",
    )

    plot_tsne_by_class(
        Xs_train_flat,
        ys_train,
        Xt_test_flat,
        yt_test,
        output_path="results/figures/branch2_tsne_class_before.png",
        title="Branch2 t-SNE (Class) - Before Adaptation",
    )

    # === AFTER ADAPTATION ===
    plot_tsne_by_domain(
        Xs_train_adapted,
        Xt_test_flat,
        output_path="results/figures/branch2_tsne_domain_after.png",
        title="Branch2 t-SNE (Domain) - After Adaptation",
    )

    plot_tsne_by_class(
        Xs_train_adapted,
        ys_train,
        Xt_test_flat,
        yt_test,
        output_path="results/figures/branch2_tsne_class_after.png",
        title="Branch2 t-SNE (Class) - After Adaptation",
    )

    print("\nt-SNE plots saved to results/figures/")

        # ===== UDA-TFL-INSPIRED =====
    # 1. Fit adaptation-aware projection using Xt_train (NO leakage)
    W_uda, mean_uda, pseudo_labels = fit_uda_tfl_inspired(
        X_source=Xs_train,
        y_source=ys_train,
        X_target=Xt_train,
        n_components=n_components,
        alpha=0.5,          # marginal weight
        beta=0.5,           # conditional weight
        n_neighbors=n_neighbors,
        n_iter=3,           # small, stable iteration count
    )

    # 2. Transform source train and target test with learned projection
    Xs_uda_proj = transform_uda_tfl_inspired(Xs_train, W=W_uda, mean_image=mean_uda)
    Xt_uda_test_proj = transform_uda_tfl_inspired(Xt_test, W=W_uda, mean_image=mean_uda)

    # 3. Flatten
    Xs_uda_flat = flatten_projected_features(Xs_uda_proj)
    Xt_uda_test_flat = flatten_projected_features(Xt_uda_test_proj)

    # 4. Also transform Xt_train for discrepancy measurement
    Xt_uda_train_proj = transform_uda_tfl_inspired(Xt_train, W=W_uda, mean_image=mean_uda)
    Xt_uda_train_flat = flatten_projected_features(Xt_uda_train_proj)

    # 5. Discrepancy (train-based, correct protocol)
    mmd_uda_train = compute_mmd(Xs_uda_flat, Xt_uda_train_flat)

    # diagnostic (test)
    mmd_uda_test = compute_mmd(Xs_uda_flat, Xt_uda_test_flat)

    # 6. Accuracy (evaluate ONLY on Xt_test)
    uda_acc = train_and_evaluate_knn(
        Xs_uda_flat,
        ys_train,
        Xt_uda_test_flat,
        yt_test,
        n_neighbors=n_neighbors,
    )


        # === UDA-TFL t-SNE ===

    # BEFORE (same baseline space for fair comparison)
    plot_tsne_by_domain(
        Xs_train_flat,
        Xt_test_flat,
        output_path="results/figures/branch2_tsne_domain_uda_before.png",
        title="UDA-TFL t-SNE (Domain) - Before",
    )

    plot_tsne_by_class(
        Xs_train_flat,
        ys_train,
        Xt_test_flat,
        yt_test,
        output_path="results/figures/branch2_tsne_class_uda_before.png",
        title="UDA-TFL t-SNE (Class) - Before",
    )

    # AFTER (UDA-TFL)
    plot_tsne_by_domain(
        Xs_uda_flat,
        Xt_uda_test_flat,
        output_path="results/figures/branch2_tsne_domain_uda_after.png",
        title="UDA-TFL t-SNE (Domain) - After",
    )

    plot_tsne_by_class(
        Xs_uda_flat,
        ys_train,
        Xt_uda_test_flat,
        yt_test,
        output_path="results/figures/branch2_tsne_class_uda_after.png",
        title="UDA-TFL t-SNE (Class) - After",
    )

    print("UDA-TFL t-SNE plots saved.")

    # === FINAL COMPARISON ===

    # PCA baseline = mevcut flat feature
    Xs_pca = Xs_train_flat
    Xt_pca = Xt_test_flat

    # CORAL
    Xs_coral = Xs_train_adapted
    Xt_coral = Xt_test_flat

    # UDA
    Xs_uda = Xs_uda_flat
    Xt_uda = Xt_uda_test_flat

    plot_comparison(
        [Xs_pca, Xs_coral, Xs_uda],
        [Xt_pca, Xt_coral, Xt_uda],
        ys_train,
        yt_test,
        titles=["PCA", "CORAL", "UDA-TFL"],
    )

    print("Final comparison figure saved!")

    # 7. Print
    print("\n=== Branch2 Experiment: UDA-TFL-Inspired ===")
    print(f"UDA MMD (train)        : {mmd_uda_train}")
    print(f"UDA MMD (test diag)    : {mmd_uda_test}")
    print(f"UDA Accuracy (Xt_test) : {uda_acc}")

    
    return {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "source_train_shape": Xs_train.shape,
        "target_train_shape": Xt_train.shape,
        "target_test_shape": Xt_test.shape,
        "source_projected_shape": Xs_train_proj.shape,
        "target_train_projected_shape": Xt_train_proj.shape,
        "target_test_projected_shape": Xt_test_proj.shape,
        "source_compact_shape": Xs_train_flat.shape,
        "target_train_compact_shape": Xt_train_flat.shape,
        "target_test_compact_shape": Xt_test_flat.shape,
        "mmd_before_train": mmd_before_train,
        "mmd_after_train": mmd_after_train,
        "mmd_before_test_diagnostic": mmd_before_test,
        "mmd_after_test_diagnostic": mmd_after_test,
        "baseline_accuracy": baseline_acc,
        "adapted_accuracy": adapted_acc,
    }


if __name__ == "__main__":
    run_branch2_experiment(n_components=4, n_neighbors=3)


