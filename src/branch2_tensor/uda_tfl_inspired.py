import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.branch2_tensor.twodpca import transform_2dpca, flatten_projected_features


def _validate_tensor_images(X, name="X"):
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 3:
        raise ValueError(
            f"{name} must be 3D with shape (n_samples, height, width), got {X.shape}"
        )

    return X


def _compute_total_scatter(X_centered):
    """
    Compute 2D-PCA-style total scatter matrix.

    X_centered shape: (n_samples, height, width)
    Output shape: (width, width)
    """
    n_samples, _, width = X_centered.shape
    scatter = np.zeros((width, width), dtype=np.float64)

    for i in range(n_samples):
        A = X_centered[i]
        scatter += A.T @ A

    return scatter / n_samples


def _compute_marginal_discrepancy(Xs_centered, Xt_centered):
    """
    Compute a simple width-mode marginal discrepancy matrix.

    This approximates source-target mean distribution mismatch
    in a 2D-PCA-compatible form.
    """
    mean_s = np.mean(Xs_centered, axis=0)
    mean_t = np.mean(Xt_centered, axis=0)

    diff = mean_s - mean_t

    return diff.T @ diff


def _compute_conditional_discrepancy(
    Xs_centered,
    y_source,
    Xt_centered,
    y_target_pseudo,
):
    """
    Compute class-wise discrepancy matrix using source labels
    and pseudo target labels.

    This is inspired by conditional distribution adaptation:
    align class-wise source and target means.
    """
    classes = np.unique(y_source)

    _, _, width = Xs_centered.shape
    discrepancy = np.zeros((width, width), dtype=np.float64)

    valid_class_count = 0

    for cls in classes:
        source_mask = y_source == cls
        target_mask = y_target_pseudo == cls

        if source_mask.sum() == 0 or target_mask.sum() == 0:
            continue

        mean_s = np.mean(Xs_centered[source_mask], axis=0)
        mean_t = np.mean(Xt_centered[target_mask], axis=0)

        diff = mean_s - mean_t
        discrepancy += diff.T @ diff
        valid_class_count += 1

    if valid_class_count > 0:
        discrepancy /= valid_class_count

    return discrepancy


def _get_top_eigenvectors(matrix, n_components):
    """
    Return top eigenvectors of a symmetric matrix.
    """
    matrix = (matrix + matrix.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]

    return eigenvectors[:, :n_components]


def _predict_pseudo_target_labels(
    Xs_features,
    y_source,
    Xt_features,
    n_neighbors=3,
):
    """
    Train KNN on source features and predict pseudo labels for target train.
    """
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(Xs_features, y_source)
    return clf.predict(Xt_features)


def fit_uda_tfl_inspired(
    X_source,
    y_source,
    X_target,
    n_components=4,
    alpha=0.5,
    beta=0.5,
    n_neighbors=3,
    n_iter=3,
):
    """
    Fit a minimal UDA-TFL-inspired projection.

    This is NOT a full reproduction of the original UDA-TFL paper.
    It is a practical, simplified, structure-preserving adaptation-aware
    projection method.

    Objective intuition:
    - preserve useful tensor variance
    - reduce marginal source-target discrepancy
    - reduce conditional discrepancy using pseudo target labels

    Parameters
    ----------
    X_source : np.ndarray
        Source train images, shape (n_source, height, width)
    y_source : np.ndarray
        Source train labels
    X_target : np.ndarray
        Target train images, unlabeled for adaptation, shape (n_target, height, width)
    n_components : int
        Number of projection components
    alpha : float
        Weight for marginal discrepancy penalty
    beta : float
        Weight for conditional discrepancy penalty
    n_neighbors : int
        KNN neighbors for pseudo-label generation
    n_iter : int
        Number of pseudo-label refinement iterations

    Returns
    -------
    W : np.ndarray
        Adaptation-aware projection matrix, shape (width, n_components)
    mean_image : np.ndarray
        Mean image used for centering
    pseudo_labels : np.ndarray
        Final pseudo labels for target train
    """
    X_source = _validate_tensor_images(X_source, name="X_source")
    X_target = _validate_tensor_images(X_target, name="X_target")
    y_source = np.asarray(y_source)

    if X_source.shape[1:] != X_target.shape[1:]:
        raise ValueError(
            f"Source and target image shapes must match, got "
            f"{X_source.shape[1:]} and {X_target.shape[1:]}"
        )

    _, _, width = X_source.shape

    if not (1 <= n_components <= width):
        raise ValueError(
            f"n_components must be between 1 and width={width}, got {n_components}"
        )

    # Use combined mean because projection is learned from both domains.
    X_all = np.concatenate([X_source, X_target], axis=0)
    mean_image = np.mean(X_all, axis=0)

    Xs_centered = X_source - mean_image
    Xt_centered = X_target - mean_image
    Xall_centered = X_all - mean_image

    # Initial projection uses total scatter only.
    total_scatter = _compute_total_scatter(Xall_centered)
    W = _get_top_eigenvectors(total_scatter, n_components=n_components)

    pseudo_labels = None

    for _ in range(n_iter):
        # Project current features.
        Xs_proj = transform_2dpca(X_source, W=W, mean_image=mean_image)
        Xt_proj = transform_2dpca(X_target, W=W, mean_image=mean_image)

        Xs_flat = flatten_projected_features(Xs_proj)
        Xt_flat = flatten_projected_features(Xt_proj)

        # Pseudo-label target train.
        pseudo_labels = _predict_pseudo_target_labels(
            Xs_flat,
            y_source,
            Xt_flat,
            n_neighbors=n_neighbors,
        )

        # Compute discrepancy penalties.
        marginal_disc = _compute_marginal_discrepancy(
            Xs_centered,
            Xt_centered,
        )

        conditional_disc = _compute_conditional_discrepancy(
            Xs_centered,
            y_source,
            Xt_centered,
            pseudo_labels,
        )

        # Adaptation-aware projection matrix.
        # We maximize variance while penalizing domain discrepancy.
        objective_matrix = (
            total_scatter
            - alpha * marginal_disc
            - beta * conditional_disc
        )

        W = _get_top_eigenvectors(objective_matrix, n_components=n_components)

    return W, mean_image, pseudo_labels


def transform_uda_tfl_inspired(X, W, mean_image):
    """
    Transform images using the learned UDA-TFL-inspired projection.
    """
    return transform_2dpca(X, W=W, mean_image=mean_image)


def fit_transform_uda_tfl_inspired(
    X_source,
    y_source,
    X_target,
    n_components=4,
    alpha=0.5,
    beta=0.5,
    n_neighbors=3,
    n_iter=3,
):
    """
    Fit UDA-TFL-inspired projection and transform source/target train.
    """
    W, mean_image, pseudo_labels = fit_uda_tfl_inspired(
        X_source=X_source,
        y_source=y_source,
        X_target=X_target,
        n_components=n_components,
        alpha=alpha,
        beta=beta,
        n_neighbors=n_neighbors,
        n_iter=n_iter,
    )

    Xs_proj = transform_uda_tfl_inspired(X_source, W=W, mean_image=mean_image)
    Xt_proj = transform_uda_tfl_inspired(X_target, W=W, mean_image=mean_image)

    return Xs_proj, Xt_proj, W, mean_image, pseudo_labels