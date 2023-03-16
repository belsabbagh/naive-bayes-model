import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

from src.nb import NB


def score_test(y_pred: np.ndarray, y_test, pos_label):
    return {
        'accuracy': round(accuracy_score(y_test, y_pred), 3),
        'f1': round(f1_score(y_test, y_pred, pos_label=pos_label), 3),
        'recall': round(recall_score(y_test, y_pred, average="binary", pos_label=pos_label), 3),
        'precision': round(precision_score(y_test, y_pred, pos_label=pos_label), 3),
    }


def test_model_kfold(model, X, y, pos_label, verbose=False):
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    scores = {'accuracy': [], 'f1': [], 'recall': [], 'precision': []}
    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = (
            X.iloc[train_index],
            X.iloc[test_index],
            y.iloc[train_index],
            y.iloc[test_index]
        )
        model.fit(X_train, y_train)
        y_pred = [model.predict(row) for index, row in X_test.iterrows()]
        results = score_test(y_pred, y_test, pos_label)
        scores = {k: scores[k] + [results[k]] for k in scores}
    return scores


def print_scores(scores):
    print('Accuracy:', '%.3f +/- %.3f' %
          (np.mean(scores['accuracy']), np.std(scores['accuracy'])))
    print('F1 Score:', '%.3f +/- %.3f' %
          (np.mean(scores['f1']), np.std(scores['f1'])))
    print('Recall:', '%.3f +/- %.3f' %
          (np.mean(scores['recall']), np.std(scores['recall'])))
    print('Precision:', '%.3f +/- %.3f' %
          (np.mean(scores['precision']), np.std(scores['precision'])))


if __name__ == '__main__':
    df = pd.read_csv('data/tennis.csv')
    label = 'Play'
    X, y = df.loc[:, df.columns != label], df[label]
    # X = X.astype(np.float32)
    # X = (df-df.mean())/df.std()
    # y = y.map({'M': 1, 'B': 0})
    scores = test_model_kfold(NB(), X, y, 'Yes')
    print_scores(scores)
