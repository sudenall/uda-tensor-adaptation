
# import numpy as np 1.ASAMA
# from src.adaptation.coral import coral

# # fake data
# Xs = np.random.randn(100, 20)
# Xt = np.random.randn(100, 20) + 2  # shifted distribution

# Xs_adapted = coral(Xs, Xt)

# print("Before adaptation mean:", np.mean(Xs))
# print("After adaptation mean:", np.mean(Xs_adapted))


""" 
 2.ASAMA

import numpy as np  
from src.adaptation.coral import coral
from src.adaptation.mmd import compute_mmd

np.random.seed(42)

# Fake source and target domains
Xs = np.random.randn(200, 20)
Xt = (np.random.randn(200, 20) * 1.5) + 3.0

# MMD before adaptation
mmd_before = compute_mmd(Xs, Xt, gamma=0.1)

# CORAL adaptation
Xs_adapted = coral(Xs, Xt)

# MMD after adaptation
mmd_after = compute_mmd(Xs_adapted, Xt, gamma=0.1)

print("MMD before adaptation:", mmd_before)
print("MMD after adaptation :", mmd_after) """


"""
3.ASAMA

import numpy as np

from src.utils.io import load_digits_domain_shift
from src.adaptation.coral import coral
from src.adaptation.mmd import compute_mmd

np.random.seed(42)

# Load source and target domains
Xs_train, Xs_test, Xt_train, Xt_test, ys_train, ys_test, yt_train, yt_test = load_digits_domain_shift()

# Flatten only for adaptation test
Xs_train_flat = Xs_train.reshape(Xs_train.shape[0], -1)
Xt_train_flat = Xt_train.reshape(Xt_train.shape[0], -1)

# Measure discrepancy before adaptation
mmd_before = compute_mmd(Xs_train_flat, Xt_train_flat, gamma=0.5)

# Apply CORAL
Xs_adapted = coral(Xs_train_flat, Xt_train_flat)

# Measure discrepancy after adaptation
mmd_after = compute_mmd(Xs_adapted, Xt_train_flat, gamma=0.5)

print("Source train shape:", Xs_train.shape)
print("Target train shape:", Xt_train.shape)
print("Flattened source shape:", Xs_train_flat.shape)
print("MMD before adaptation:", mmd_before)
print("MMD after adaptation :", mmd_after)"""


"""
4.ASAMA

import numpy as np

from src.utils.io import load_digits_domain_shift
from src.branch1_vector.preprocessing import validate_image_array, flatten_images
from src.adaptation.coral import coral
from src.adaptation.mmd import compute_mmd

np.random.seed(42)

# Load source and target domains
Xs_train, Xs_test, Xt_train, Xt_test, ys_train, ys_test, yt_train, yt_test = load_digits_domain_shift()

# Validate image tensors
validate_image_array(Xs_train)
validate_image_array(Xt_train)

# Branch1 preprocessing: flatten images into vectors
Xs_train_flat = flatten_images(Xs_train)
Xt_train_flat = flatten_images(Xt_train)

# Measure discrepancy before adaptation
mmd_before = compute_mmd(Xs_train_flat, Xt_train_flat, gamma=0.5)

# Apply CORAL
Xs_adapted = coral(Xs_train_flat, Xt_train_flat)

# Measure discrepancy after adaptation
mmd_after = compute_mmd(Xs_adapted, Xt_train_flat, gamma=0.5)

print("Source train shape:", Xs_train.shape)
print("Target train shape:", Xt_train.shape)
print("Flattened source shape:", Xs_train_flat.shape)
print("MMD before adaptation:", mmd_before)
print("MMD after adaptation :", mmd_after)"""


"""
5.ASAMA

import numpy as np

from src.utils.io import load_digits_domain_shift
from src.branch1_vector.preprocessing import validate_image_array, flatten_images
from src.branch1_vector.pca_pipeline import fit_pca, transform_with_pca
from src.adaptation.coral import coral
from src.adaptation.mmd import compute_mmd

np.random.seed(42)

# Load source and target domains
Xs_train, Xs_test, Xt_train, Xt_test, ys_train, ys_test, yt_train, yt_test = load_digits_domain_shift()

# Validate image tensors
validate_image_array(Xs_train)
validate_image_array(Xt_train)

# Branch1 preprocessing: flatten images
Xs_train_flat = flatten_images(Xs_train)
Xt_train_flat = flatten_images(Xt_train)

# Fit PCA on source train only
pca = fit_pca(Xs_train_flat, n_components=20)

# Transform source and target into PCA space
Xs_train_pca = transform_with_pca(pca, Xs_train_flat)
Xt_train_pca = transform_with_pca(pca, Xt_train_flat)

# Measure discrepancy before adaptation in PCA space
mmd_before = compute_mmd(Xs_train_pca, Xt_train_pca, gamma=0.5)

# Apply CORAL in PCA space
Xs_adapted = coral(Xs_train_pca, Xt_train_pca)

# Measure discrepancy after adaptation
mmd_after = compute_mmd(Xs_adapted, Xt_train_pca, gamma=0.5)

print("Source train shape:", Xs_train.shape)
print("Target train shape:", Xt_train.shape)
print("Flattened source shape:", Xs_train_flat.shape)
print("PCA source shape:", Xs_train_pca.shape)
print("MMD before adaptation:", mmd_before)
print("MMD after adaptation :", mmd_after)"""


"""
6.ASAMA


import numpy as np

from src.utils.io import load_digits_domain_shift
from src.branch1_vector.preprocessing import validate_image_array, flatten_images
from src.branch1_vector.pca_pipeline import fit_pca, transform_with_pca
from src.branch1_vector.classifier import train_knn_classifier, evaluate_classifier
from src.adaptation.coral import coral
from src.adaptation.mmd import compute_mmd

np.random.seed(42)

# Load source and target domains
Xs_train, Xs_test, Xt_train, Xt_test, ys_train, ys_test, yt_train, yt_test = load_digits_domain_shift()

# Validate image tensors
validate_image_array(Xs_train)
validate_image_array(Xt_train)
validate_image_array(Xs_test)
validate_image_array(Xt_test)

# Branch1 preprocessing: flatten
Xs_train_flat = flatten_images(Xs_train)
Xt_train_flat = flatten_images(Xt_train)
Xs_test_flat = flatten_images(Xs_test)
Xt_test_flat = flatten_images(Xt_test)

# Fit PCA on source train only
pca = fit_pca(Xs_train_flat, n_components=20)

# Transform all splits into PCA space
Xs_train_pca = transform_with_pca(pca, Xs_train_flat)
Xt_train_pca = transform_with_pca(pca, Xt_train_flat)
Xs_test_pca = transform_with_pca(pca, Xs_test_flat)
Xt_test_pca = transform_with_pca(pca, Xt_test_flat)

# MMD before adaptation
mmd_before = compute_mmd(Xs_train_pca, Xt_train_pca, gamma=0.5)

# Train classifier without adaptation
baseline_model = train_knn_classifier(Xs_train_pca, ys_train)
baseline_acc = evaluate_classifier(baseline_model, Xt_test_pca, yt_test)

# Apply CORAL on source train and source test
Xs_train_adapted = coral(Xs_train_pca, Xt_train_pca)
Xs_test_adapted = coral(Xs_test_pca, Xt_train_pca)

# MMD after adaptation
mmd_after = compute_mmd(Xs_train_adapted, Xt_train_pca, gamma=0.5)

# Train classifier with adapted source
adapted_model = train_knn_classifier(Xs_train_adapted, ys_train)
adapted_acc = evaluate_classifier(adapted_model, Xt_test_pca, yt_test)

print("Source train shape:", Xs_train.shape)
print("Target test shape:", Xt_test.shape)
print("PCA source shape:", Xs_train_pca.shape)
print("MMD before adaptation:", mmd_before)
print("MMD after adaptation :", mmd_after)
print("Baseline accuracy on target:", baseline_acc)
print("Adapted accuracy on target :", adapted_acc)"""

"""
7.ASAMA
import numpy as np

from src.utils.io import load_digits_domain_shift
from src.utils.results import save_metrics
from src.branch1_vector.preprocessing import validate_image_array, flatten_images
from src.branch1_vector.pca_pipeline import fit_pca, transform_with_pca
from src.branch1_vector.classifier import train_knn_classifier, evaluate_classifier
from src.adaptation.coral import coral
from src.adaptation.mmd import compute_mmd

np.random.seed(42)

# Load source and target domains
Xs_train, Xs_test, Xt_train, Xt_test, ys_train, ys_test, yt_train, yt_test = load_digits_domain_shift()

# Validate image tensors
validate_image_array(Xs_train)
validate_image_array(Xt_train)
validate_image_array(Xs_test)
validate_image_array(Xt_test)

# Branch1 preprocessing: flatten
Xs_train_flat = flatten_images(Xs_train)
Xt_train_flat = flatten_images(Xt_train)
Xs_test_flat = flatten_images(Xs_test)
Xt_test_flat = flatten_images(Xt_test)

# Fit PCA on source train only
pca = fit_pca(Xs_train_flat, n_components=20)

# Transform all splits into PCA space
Xs_train_pca = transform_with_pca(pca, Xs_train_flat)
Xt_train_pca = transform_with_pca(pca, Xt_train_flat)
Xs_test_pca = transform_with_pca(pca, Xs_test_flat)
Xt_test_pca = transform_with_pca(pca, Xt_test_flat)

# MMD before adaptation
mmd_before = compute_mmd(Xs_train_pca, Xt_train_pca, gamma=0.5)

# Train classifier without adaptation
baseline_model = train_knn_classifier(Xs_train_pca, ys_train)
baseline_acc = evaluate_classifier(baseline_model, Xt_test_pca, yt_test)

# Apply CORAL
Xs_train_adapted = coral(Xs_train_pca, Xt_train_pca)
Xs_test_adapted = coral(Xs_test_pca, Xt_train_pca)

# MMD after adaptation
mmd_after = compute_mmd(Xs_train_adapted, Xt_train_pca, gamma=0.5)

# Train classifier with adapted source
adapted_model = train_knn_classifier(Xs_train_adapted, ys_train)
adapted_acc = evaluate_classifier(adapted_model, Xt_test_pca, yt_test)

# Save metrics
metrics = {
    "source_train_shape": list(Xs_train.shape),
    "target_test_shape": list(Xt_test.shape),
    "pca_source_shape": list(Xs_train_pca.shape),
    "mmd_before_adaptation": float(mmd_before),
    "mmd_after_adaptation": float(mmd_after),
    "baseline_accuracy_on_target": float(baseline_acc),
    "adapted_accuracy_on_target": float(adapted_acc),
}

save_metrics(metrics, "results/metrics/branch1_metrics.json")

# Print results
print("Source train shape:", Xs_train.shape)
print("Target test shape:", Xt_test.shape)
print("PCA source shape:", Xs_train_pca.shape)
print("MMD before adaptation:", mmd_before)
print("MMD after adaptation :", mmd_after)
print("Baseline accuracy on target:", baseline_acc)
print("Adapted accuracy on target :", adapted_acc)
print("Metrics saved to: results/metrics/branch1_metrics.json")"""

import numpy as np

from src.utils.io import load_digits_domain_shift
from src.utils.results import save_metrics
from src.branch1_vector.preprocessing import validate_image_array, flatten_images
from src.branch1_vector.pca_pipeline import fit_pca, transform_with_pca
from src.branch1_vector.classifier import train_knn_classifier, evaluate_classifier
from src.adaptation.coral import coral
from src.adaptation.mmd import compute_mmd
from src.visualization.metrics_plots import plot_branch1_summary
from src.visualization.confusion_matrix_plot import plot_confusion_matrix
from src.branch1_vector.classifier import train_logreg
from src.branch1_vector.classifier import train_svm
import pandas as pd
from pathlib import Path

np.random.seed(42)

# Load source and target domains
Xs_train, Xs_test, Xt_train, Xt_test, ys_train, ys_test, yt_train, yt_test = load_digits_domain_shift()

# Validate image tensors
validate_image_array(Xs_train)
validate_image_array(Xt_train)
validate_image_array(Xs_test)
validate_image_array(Xt_test)

# Branch1 preprocessing: flatten
Xs_train_flat = flatten_images(Xs_train)
Xt_train_flat = flatten_images(Xt_train)
Xs_test_flat = flatten_images(Xs_test)
Xt_test_flat = flatten_images(Xt_test)

# Fit PCA on source train only
#pca = fit_pca(Xs_train_flat, n_components=20)
n_components = 40
pca = fit_pca(Xs_train_flat, n_components=n_components)

# Transform all splits into PCA space
Xs_train_pca = transform_with_pca(pca, Xs_train_flat)
Xt_train_pca = transform_with_pca(pca, Xt_train_flat)
Xs_test_pca = transform_with_pca(pca, Xs_test_flat)
Xt_test_pca = transform_with_pca(pca, Xt_test_flat)

# MMD before adaptation
mmd_before = compute_mmd(Xs_train_pca, Xt_train_pca, gamma=0.5)

k = 1 # bunu değiştirerek deneyeceğiz
print("KNN neighbors:", k)

# Train classifier without adaptation
baseline_model = train_knn_classifier(Xs_train_pca, ys_train, n_neighbors=k)
#baseline_acc = evaluate_classifier(baseline_model, Xt_test_pca, yt_test)
baseline_acc, baseline_preds, baseline_report = evaluate_classifier(
    baseline_model, Xt_test_pca, yt_test,
    return_predictions=True,
    return_report=True
)

#logistik regresyonla deniyoruz bu sefer
# Logistic Regression
# Logistic Regression baseline
logreg_model = train_logreg(Xs_train_pca, ys_train)
logreg_acc = evaluate_classifier(logreg_model, Xt_test_pca, yt_test)

print("LogReg baseline:", logreg_acc)

#şimdi de svm ile deniyoruz
svm_model = train_svm(Xs_train_pca, ys_train)
svm_acc = evaluate_classifier(svm_model, Xt_test_pca, yt_test)

print("SVM baseline:", svm_acc)

# Apply CORAL
Xs_train_adapted = coral(Xs_train_pca, Xt_train_pca)
Xs_test_adapted = coral(Xs_test_pca, Xt_train_pca)

# MMD after adaptation
mmd_after = compute_mmd(Xs_train_adapted, Xt_train_pca, gamma=0.5)

# Train classifier with adapted source
adapted_model = train_knn_classifier(Xs_train_adapted, ys_train, n_neighbors=k)
#adapted_acc = evaluate_classifier(adapted_model, Xt_test_pca, yt_test)
adapted_acc, adapted_preds, adapted_report = evaluate_classifier(
    adapted_model, Xt_test_pca, yt_test,
    return_predictions=True,
    return_report=True
)

# Logistic Regression adapted
logreg_adapted_model = train_logreg(Xs_train_adapted, ys_train)
logreg_adapted_acc = evaluate_classifier(logreg_adapted_model, Xt_test_pca, yt_test)

print("LogReg adapted :", logreg_adapted_acc)

#svm adapted
svm_adapted_model = train_svm(Xs_train_adapted, ys_train)
svm_adapted_acc = evaluate_classifier(svm_adapted_model, Xt_test_pca, yt_test)

print("SVM adapted :", svm_adapted_acc)


# Save metrics
metrics = {
    "source_train_shape": list(Xs_train.shape),
    "target_test_shape": list(Xt_test.shape),
    "pca_source_shape": list(Xs_train_pca.shape),
    "mmd_before_adaptation": float(mmd_before),
    "mmd_after_adaptation": float(mmd_after),
    "baseline_accuracy_on_target": float(baseline_acc),
    "adapted_accuracy_on_target": float(adapted_acc),
    "baseline_report": baseline_report,
    "adapted_report": adapted_report,
    "logreg_baseline" : float(logreg_acc),
    "logreg_adapted" : float(logreg_adapted_acc),
    "svm_baseline" : float(svm_acc),
    "svm_adapted" : float(svm_adapted_acc)
}

metrics_path = "results/metrics/branch1_metrics.json"
figure_path = "results/figures/branch1_summary.png"

save_metrics(metrics, metrics_path)
plot_branch1_summary(metrics_path, figure_path)



baseline_cm_path = "results/figures/branch1_confusion_baseline.png"
adapted_cm_path = "results/figures/branch1_confusion_adapted.png"

plot_confusion_matrix(
    yt_test,
    baseline_preds,
    baseline_cm_path,
    title="Baseline Confusion Matrix"
)

plot_confusion_matrix(
    yt_test,
    adapted_preds,
    adapted_cm_path,
    title="Adapted Confusion Matrix"
)

# Print results

print("Source train shape:", Xs_train.shape)
print("Target test shape:", Xt_test.shape)
print("PCA source shape:", Xs_train_pca.shape)
print("PCA n_components:", n_components)

print("MMD before adaptation:", mmd_before)
print("MMD after adaptation :", mmd_after)
print("Baseline accuracy on target:", baseline_acc)
print("Adapted accuracy on target :", adapted_acc)
print(f"Metrics saved to: {metrics_path}")
print(f"Figure saved to : {figure_path}")

print(f"Baseline confusion matrix saved to: {baseline_cm_path}")
print(f"Adapted confusion matrix saved to : {adapted_cm_path}")



RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

comparison_data = [
    {
        "classifier": "KNN",
        "baseline_accuracy": baseline_acc,
        "adapted_accuracy": adapted_acc,
    },
    {
        "classifier": "LogisticRegression",
        "baseline_accuracy": logreg_acc,
        "adapted_accuracy": logreg_adapted_acc,
    },
    {
        "classifier": "SVM",
        "baseline_accuracy": svm_acc,
        "adapted_accuracy": svm_adapted_acc,
    },
]

df = pd.DataFrame(comparison_data)

output_path = RESULTS_DIR / "classifier_comparison.csv"
df.to_csv(output_path, index=False)

print("Classifier comparison saved to:", output_path)
print(df)