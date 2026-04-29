import numpy as np


def fit_2dpca(X, n_components):
    """
    Fit a simple column-direction 2D-PCA projection matrix.

    Parameters
    ----------
    X : np.ndarray
        Input tensor images with shape (n_samples, height, width)
    n_components : int
        Number of projection components to keep

    Returns
    -------
    W : np.ndarray
        Projection matrix with shape (width, n_components)
    mean_image : np.ndarray
        Mean image with shape (height, width)
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 3:
        raise ValueError(
            f"X must be 3D with shape (n_samples, height, width), got {X.shape}"
        )

    n_samples, height, width = X.shape

    if not (1 <= n_components <= width):
        raise ValueError(
            f"n_components must be between 1 and width={width}, got {n_components}"
        )

    mean_image = np.mean(X, axis=0)

    G = np.zeros((width, width), dtype=np.float64)

    for i in range(n_samples):
        A = X[i] - mean_image
        G += A.T @ A

    G /= n_samples

    eigenvalues, eigenvectors = np.linalg.eigh(G)

    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    eigenvalues = eigenvalues[sorted_idx]

    W = eigenvectors[:, :n_components]

    return W, mean_image


def transform_2dpca(X, W, mean_image):
    """
    Project images into 2D-PCA subspace.

    Parameters
    ----------
    X : np.ndarray
        Input tensor images with shape (n_samples, height, width)
    W : np.ndarray
        Projection matrix with shape (width, n_components)
    mean_image : np.ndarray
        Mean image with shape (height, width)

    Returns
    -------
    X_proj : np.ndarray
        Projected tensor features with shape (n_samples, height, n_components)
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 3:
        raise ValueError(
            f"X must be 3D with shape (n_samples, height, width), got {X.shape}"
        )

    n_samples, height, width = X.shape

    if mean_image.shape != (height, width):
        raise ValueError(
            f"mean_image shape mismatch: expected {(height, width)}, got {mean_image.shape}"
        )

    if W.shape[0] != width:
        raise ValueError(
            f"W shape mismatch: expected first dim {width}, got {W.shape[0]}"
        )

    projected = []
    for i in range(n_samples):
        A = X[i] - mean_image
        Y = A @ W
        projected.append(Y)

    return np.stack(projected, axis=0)


def flatten_projected_features(X_proj):
    """
    Flatten projected 2D-PCA tensor features into compact vectors.

    Parameters
    ----------
    X_proj : np.ndarray
        Projected features with shape (n_samples, height, n_components)

    Returns
    -------
    X_flat : np.ndarray
        Flattened compact features with shape (n_samples, height * n_components)
    """
    X_proj = np.asarray(X_proj, dtype=np.float64)

    if X_proj.ndim != 3:
        raise ValueError(
            f"X_proj must be 3D with shape (n_samples, height, n_components), got {X_proj.shape}"
        )

    n_samples = X_proj.shape[0]
    return X_proj.reshape(n_samples, -1)


def fit_transform_2dpca(X_train, n_components):
    """
    Fit 2D-PCA on training data and return projected train features.
    """
    W, mean_image = fit_2dpca(X_train, n_components=n_components)
    X_train_proj = transform_2dpca(X_train, W=W, mean_image=mean_image)
    return X_train_proj, W, mean_image


def summarize_projected_features(X_proj, name="X_proj"):
    """
    Return simple summary information for projected tensor features.
    """
    X_proj = np.asarray(X_proj)

    if X_proj.ndim != 3:
        raise ValueError(
            f"{name} must be 3D with shape (n_samples, height, n_components), got {X_proj.shape}"
        )

    return {
        "name": name,
        "shape": X_proj.shape,
        "dtype": str(X_proj.dtype),
        "min": float(X_proj.min()),
        "max": float(X_proj.max()),
        "mean": float(X_proj.mean()),
        "std": float(X_proj.std()),
    }


def print_projected_summary(X_proj, name="X_proj"):
    """
    Print a simple debug summary for projected tensor features.
    """
    summary = summarize_projected_features(X_proj, name=name)
    print(
        f"{summary['name']} summary -> "
        f"shape: {summary['shape']}, "
        f"dtype: {summary['dtype']}, "
        f"min: {summary['min']:.4f}, "
        f"max: {summary['max']:.4f}, "
        f"mean: {summary['mean']:.4f}, "
        f"std: {summary['std']:.4f}"
    )