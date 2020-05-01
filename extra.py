from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, GridSearchCV
import matplotlib.pyplot as plt
from itertools import combinations

K = [1, 3, 5, 7]
c_range = list(2.**np.arange(-5, 15, 2))
gamma_range = list(2.**np.arange(-15, 3, 2))


def knn_accuracy(X_train, X_test, y_train, y_test):
    params = {"n_neighbors": K}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=params, n_jobs=-1, scoring="accuracy", cv=5)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    return grid.best_params_, accuracy_score(y_test, y_pred)


def svm_linear_accuracy(X_train, X_test, y_train, y_test):
    params = {
        "C": c_range,
        "gamma": gamma_range
    }
    grid = GridSearchCV(SVC(kernel="linear"), param_grid=params, n_jobs=-1, scoring="accuracy", cv=5)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    return grid.best_params_, accuracy_score(y_test, y_pred)


def svm_rbf_accuracy(X_train, X_test, y_train, y_test):
    params = {
        "C": c_range,
        "gamma": gamma_range
    }
    grid = GridSearchCV(SVC(kernel="rbf"), param_grid=params, n_jobs=-1, scoring="accuracy", cv=5)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    return grid.best_params_, accuracy_score(y_test, y_pred)


def start_extra_task():
    dataset = load_wine()
    y_true = dataset.target
    comb = list(combinations(range(len(dataset.feature_names)), 2))

    accuracies = {}
    for features in comb[:3]:
        X = dataset.data[:, features]
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=3/10, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        args = [X_train, X_test, y_train, y_test]

        accuracies[features] = {}
        accuracies[features]["knn"] = knn_accuracy(*args)
        accuracies[features]['svm-linear'] = svm_linear_accuracy(*args)
        accuracies[features]['svm-rbf'] = svm_rbf_accuracy(*args)

    return accuracies


if __name__ == '__main__':
    accuracies = start_extra_task()
    for item in list(accuracies.items()):
        print(item)