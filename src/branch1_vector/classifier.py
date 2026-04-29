from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def train_knn_classifier(X_train, y_train, n_neighbors=3):
    """
    Train a KNN classifier on feature vectors.

    Parameters:
        X_train: np.ndarray of shape (n_samples, n_features)
        y_train: np.ndarray of shape (n_samples,)
        n_neighbors: int

    Returns:
        trained KNN classifier
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

"""
def evaluate_classifier(model, X_test, y_test):
    
    Evaluate a classifier and return accuracy.

    Parameters:
        model: trained classifier
        X_test: np.ndarray
        y_test: np.ndarray

    Returns:
        float: accuracy score
    
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def evaluate_classifier(model, X_test, y_test, return_predictions=False):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if return_predictions:
        return acc, y_pred

    return acc"""

def evaluate_classifier(model, X_test, y_test, return_predictions=False, return_report=False):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = None
    if return_report:
        report = classification_report(y_test, y_pred, output_dict=True)

    if return_predictions and return_report:
        return acc, y_pred, report
    elif return_predictions:
        return acc, y_pred
    elif return_report:
        return acc, report

    return acc

def train_logreg(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    return model