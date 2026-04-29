import numpy as np


def validate_tensor_images(X, name="X"):
    """
    Validate that input is a 3D image tensor array with shape:
    (n_samples, height, width)

    Parameters
    ----------
    X : array-like
        Input image tensor array.
    name : str
        Name used in error messages.

    Returns
    -------
    X : np.ndarray
        Validated numpy array.
    """
    X = np.asarray(X)

    if X.ndim != 3:
        raise ValueError(
            f"{name} must be a 3D array with shape "
            f"(n_samples, height, width), but got shape {X.shape}"
        )

    n_samples, height, width = X.shape

    if n_samples == 0:
        raise ValueError(f"{name} must contain at least 1 sample.")

    if height <= 0 or width <= 0:
        raise ValueError(
            f"{name} has invalid spatial dimensions: height={height}, width={width}"
        )

    return X


def prepare_tensor_images(X, name="X", normalize=True):
    """
    Prepare image tensors for Branch2 pipeline.

    Steps:
    - validate shape
    - convert to float64
    - optionally normalize to [0, 1] if values are outside that range

    Parameters
    ----------
    X : array-like
        Input tensor images with shape (n_samples, height, width)
    name : str
        Name used in validation messages
    normalize : bool
        Whether to normalize values if needed

    Returns
    -------
    X_prepared : np.ndarray
        Prepared tensor images with dtype float64
    """
    X = validate_tensor_images(X, name=name).astype(np.float64)

    if normalize:
        x_min = X.min()
        x_max = X.max()

        # Only normalize if data is not already in [0, 1]
        if x_min < 0.0 or x_max > 1.0:
            if np.isclose(x_max, x_min):
                raise ValueError(
                    f"{name} cannot be normalized because all values are constant: {x_min}"
                )
            X = (X - x_min) / (x_max - x_min)

    return X


def summarize_tensor_images(X, name="X"):
    """
    Return a lightweight summary dictionary for debugging/logging.
    """
    X = validate_tensor_images(X, name=name)

    return {
        "name": name,
        "shape": X.shape,
        "dtype": str(X.dtype),
        "min": float(X.min()),
        "max": float(X.max()),
        "mean": float(X.mean()),
        "std": float(X.std()),
    }


def print_tensor_summary(X, name="X"):
    """
    Print a simple summary for quick debugging.
    """
    summary = summarize_tensor_images(X, name=name)
    print(
        f"{summary['name']} summary -> "
        f"shape: {summary['shape']}, "
        f"dtype: {summary['dtype']}, "
        f"min: {summary['min']:.4f}, "
        f"max: {summary['max']:.4f}, "
        f"mean: {summary['mean']:.4f}, "
        f"std: {summary['std']:.4f}"
    )