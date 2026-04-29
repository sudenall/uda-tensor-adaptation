#MMD measures the adaptation effect.
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def compute_mmd(X, Y, gamma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between two sets of features.

    Parameters:
        X (np.ndarray): source features, shape (n_samples_x, n_features)
        Y (np.ndarray): target features, shape (n_samples_y, n_features)
        gamma (float): RBF kernel parameter

    Returns:
        float: MMD score
    """
    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return float(mmd)