import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from parameter_selection import ParameterSelectionHelper


def main():
    in_file = sys.argv[1]
    out_file = sys.argv[2]

    df = pd.read_csv(in_file)

    X = df.iloc[:, :2].as_matrix()
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)

    models1 = {
        'svm_linear': SVC(kernel='linear'),
        'svm_polynomial': SVC(kernel='poly'),
        'svm_rbf': SVC(kernel='rbf'),
        'logistic': LogisticRegression(),
        'knn': KNeighborsClassifier(),
        'decision_tree': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier()
    }

    params1 = {
        'svm_linear': {'C': [0.1, 0.5, 1, 5, 10, 50, 100]},
        'svm_polynomial': {'C': [0.1, 1, 3], 'gamma': [0.1, 0.5], 'degree': [4, 5, 6]},
        'svm_rbf': {'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10]},
        'logistic': {'C': [0.1, 0.5, 1, 5, 10, 50, 100]},
        'knn': {'n_neighbors': range(1, 50), 'leaf_size': range(5, 60, 5)},
        'decision_tree': {'max_depth': range(1, 50), 'min_samples_split': range(2, 10, 1)},
        'random_forest': {'max_depth': range(1, 50), 'min_samples_split': range(2, 10, 1)}
    }

    file = open(out_file, 'w+')
    for key in models1.keys():
        helper = ParameterSelectionHelper(models1, params1)
        gridsearch = helper.fit(X_train, y_train, key)
        y_train_pred = gridsearch.predict(X_train)
        y_test_pred = gridsearch.predict(X_test)
        test_score = accuracy_score(y_test_pred, y_test)
        best_score = accuracy_score(y_train_pred, y_train)

        file.write(
            (key +
             ',' +
             str(best_score) +
                ',' +
                str(test_score) +
                '\n'))
    print("Scores saved to " + out_file + " successfully.")


if __name__ == "__main__":
    main()
