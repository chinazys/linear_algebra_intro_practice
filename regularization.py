# Please, compare and analyze results. Add conclusions as comments here or to a readme file.

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import make_scorer, r2_score, accuracy_score


def preprocess(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """
    Preprocesses the input data by scaling features and splitting into training and test sets.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    return [X_train, X_test, y_train, y_test]


def get_regression_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the diabetes dataset for regression tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_diabetes()
    X, y = data.data, data.target
    return preprocess(X, y)


def get_classification_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the breast cancer dataset for classification tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    return preprocess(X, y)


def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a linear regression model on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a ridge regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best ridge regression model found by GridSearchCV.
    """
    ridge = Ridge(random_state=0)
    param_grid = {"alpha": np.logspace(-3, 3, 20)}
    gs = GridSearchCV(
        ridge,
        param_grid=param_grid,
        scoring=make_scorer(r2_score),
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X, y)
    return gs.best_estimator_


def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a lasso regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best lasso regression model found by GridSearchCV.
    """
    lasso = Lasso(max_iter=10000, random_state=0)
    param_grid = {"alpha": np.logspace(-3, 3, 20)}
    gs = GridSearchCV(
        lasso,
        param_grid=param_grid,
        scoring=make_scorer(r2_score),
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X, y)
    return gs.best_estimator_


def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model without regularization on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained logistic regression model.
    """
    logreg = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=10000,
        random_state=0,
    )
    logreg.fit(X, y)
    return logreg


def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L2 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L2 regularization found by GridSearchCV.
    """
    base = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=10000,
        random_state=0,
    )
    param_grid = {"C": np.logspace(-4, 4, 15)}
    gs = GridSearchCV(
        base,
        param_grid=param_grid,
        scoring=make_scorer(accuracy_score),
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X, y)
    return gs.best_estimator_


def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L1 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L1 regularization found by GridSearchCV.
    """
    base = LogisticRegression(
        penalty="l1",
        solver="liblinear",  # or 'saga' for larger datasets / multinomial
        max_iter=10000,
        random_state=0,
    )
    param_grid = {"C": np.logspace(-4, 4, 15)}
    gs = GridSearchCV(
        base,
        param_grid=param_grid,
        scoring=make_scorer(accuracy_score),
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X, y)
    return gs.best_estimator_
