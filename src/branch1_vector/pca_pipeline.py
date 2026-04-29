from sklearn.decomposition import PCA


def fit_pca(X_train, n_components=20, random_state=42):
    """
    Fit PCA on training data.

    Parameters:
        X_train: np.ndarray of shape (n_samples, n_features)
        n_components: number of principal components

    Returns:
        pca: fitted PCA object
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_train)
    return pca


def transform_with_pca(pca, X):
    """
    Transform data using a fitted PCA model.

    Parameters:
        pca: fitted PCA object
        X: np.ndarray of shape (n_samples, n_features)

    Returns:
        transformed array
    """
    return pca.transform(X)