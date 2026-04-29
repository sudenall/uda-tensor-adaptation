import numpy as np


def flatten_images(X):
    """
    Flatten image tensors into feature vectors.

    Parameters:
        X (np.ndarray): image array of shape (n_samples, height, width)

    Returns:
        np.ndarray: flattened array of shape (n_samples, height * width)
    """
    return X.reshape(X.shape[0], -1)


def validate_image_array(X):
    """
    Validate that the input is a 3D image array.

    Parameters:
        X (np.ndarray): expected shape (n_samples, height, width)

    Raises:
        ValueError: if input shape is invalid
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if X.ndim != 3:
        raise ValueError(
            f"Expected a 3D array with shape (n_samples, height, width), got shape {X.shape}"
        )