#What does CORAL (Correlation Alignment) do? It equalizes the covariance matrices of the source and target features.In other words: "Make the distributions similar to each other"
#applies adaptation

import numpy as np


def compute_covariance(X):
    """
    Compute covariance matrix of centered features.

    Parameters:
        X (np.ndarray): shape (n_samples, n_features)

    Returns:
        np.ndarray: covariance matrix
    """
    return np.cov(X, rowvar=False)


def coral(source, target):
    """
    Perform CORAL domain adaptation by aligning source covariance
    to target covariance.

    Parameters:
        source (np.ndarray): source features, shape (Ns, d)
        target (np.ndarray): target features, shape (Nt, d)

    Returns:
        np.ndarray: adapted source features
    """
    # Means
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)

    # Center data
    source_centered = source - source_mean
    target_centered = target - target_mean

    # Covariances with regularization
    Cs = compute_covariance(source_centered) + np.eye(source.shape[1])
    Ct = compute_covariance(target_centered) + np.eye(target.shape[1])

    # Source whitening
    U_s, S_s, _ = np.linalg.svd(Cs)
    Cs_inv_sqrt = U_s @ np.diag(1.0 / np.sqrt(S_s)) @ U_s.T

    # Target coloring
    U_t, S_t, _ = np.linalg.svd(Ct)
    Ct_sqrt = U_t @ np.diag(np.sqrt(S_t)) @ U_t.T

    # Align source to target
    adapted_source = source_centered @ Cs_inv_sqrt @ Ct_sqrt

    # Add target mean back
    adapted_source = adapted_source + target_mean

    return adapted_source