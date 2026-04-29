from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_knn_classifier(X_train, y_train, n_neighbors=3):
    """
    Train a KNN classifier on compact Branch2 features.
    """
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    return clf


def predict_classifier(clf, X_test):
    """
    Generate predictions using a trained classifier.
    """
    return clf.predict(X_test)


def evaluate_classifier(y_true, y_pred, return_report=False):
    """
    Evaluate predictions with accuracy, optionally classification report.
    """
    acc = accuracy_score(y_true, y_pred)

    if return_report:
        report = classification_report(y_true, y_pred, output_dict=True)
        return acc, report

    return acc


def train_and_evaluate_knn(
    X_train,
    y_train,
    X_test,
    y_test,
    n_neighbors=3,
    return_predictions=False,
    return_report=False,
):
    """
    Full helper:
    - train KNN
    - predict on test
    - return accuracy
    - optionally return predictions and/or report
    """
    clf = train_knn_classifier(X_train, y_train, n_neighbors=n_neighbors)
    y_pred = predict_classifier(clf, X_test)

    if return_predictions and return_report:
        acc, report = evaluate_classifier(y_test, y_pred, return_report=True)
        return acc, y_pred, report

    if return_predictions:
        acc = evaluate_classifier(y_test, y_pred, return_report=False)
        return acc, y_pred

    if return_report:
        acc, report = evaluate_classifier(y_test, y_pred, return_report=True)
        return acc, report

    acc = evaluate_classifier(y_test, y_pred, return_report=False)
    return acc