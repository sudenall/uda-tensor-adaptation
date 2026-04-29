import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_digits_domain_shift(test_size=0.3, random_state=42):
    """
    Load sklearn digits dataset and create an artificial domain shift.

    Source domain: original images
    Target domain: noisy + slightly shifted version

    Returns:
        Xs_train, Xs_test, Xt_train, Xt_test, ys_train, ys_test, yt_train, yt_test
    """
    np.random.seed(random_state)

    digits = load_digits()

    X = digits.images  # shape: (n_samples, 8, 8)
    y = digits.target

    # Normalize to [0, 1]
    X = X / 16.0

    # Source domain = original images
    X_source = X.copy()

    # Target domain = shifted + noisy version
    noise = np.random.normal(0, 0.08, X.shape)
    X_target = np.clip(np.roll(X, shift=1, axis=2) + noise, 0.0, 1.0)

    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_source,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    Xt_train, Xt_test, yt_train, yt_test = train_test_split(
        X_target,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return Xs_train, Xs_test, Xt_train, Xt_test, ys_train, ys_test, yt_train, yt_test