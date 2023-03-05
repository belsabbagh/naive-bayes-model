import numpy as np
import pandas as pd

class ModelNotFittedException(Exception):
    def __init__(self):
        super().__init__('Model not fitted')


class NB(object):
    def __init__(self, alpha=1):
        self.prob = None
        self.labels = None
        self.alpha = alpha

    @staticmethod
    def _cp(x, n, k, alpha):
        return (x + alpha)/(n + k*alpha)

    def _cond_prob(self, df: pd.DataFrame, col: str, target: str) -> dict[tuple[str, str], float]:
        """Calculates conditional probability for a dataframe column with respect to a target column

        Args:
            df (pd.DataFrame): The entire dataframe. (must have X and y)
            col (str): The X column name.
            target (str): The y column name.

        Returns:
            float: The conditional probability of occurrence for each value in the column.
        """

        def calc(g):
            return self._cp(g[g[target] == l].shape[0], g.shape[0], len(self.labels), self.alpha)
        prob = {}
        for l in self.labels:
            for v, g in df.groupby(col):
                prob[(v, l)] = calc(g)
        return prob

    def fit(self, X: pd.DataFrame, y):
        self.labels = y.unique()
        self.prob = {col: self._cond_prob(
            pd.concat([X, y], axis=1), col, y.name) for col in X.columns}

    def predict(self, X):
        if self.prob is None or self.labels is None:
            raise ModelNotFittedException()

        def p(X, lvalue):
            res = 1
            for col, val in X.items():
                res *= self.prob[col].get((val, lvalue), )
            return res
        label_probs = {l: p(X.to_dict(), l) for l in self.labels}
        print(label_probs)
        return max(label_probs, key=label_probs.get)


class GaussianNB(NB):
    def __init__(self):
        super().__init__(None)
        self.classes = None
        self.priors = {}
        self.means = {}
        self.covs = {}

    def fit(self, X, t):
        self.classes = np.unique(t)
        for c in self.classes:
            X_c = X[t == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.means[c] = np.mean(X_c, axis=0)
            self.covs[c] = np.diag(np.diag(np.cov(X_c, rowvar=False)))
            
    def predict(self, X):
        if self.classes is None:
            raise ModelNotFittedException()
        preds = []
        for x in X:
            posts = list()
            for c in self.classes:
                prior = np.log(self.priors[c])
                inv_cov = np.linalg.inv(self.covs[c])
                inv_cov_det = np.linalg.det(inv_cov)
                diff = x-self.means[c]
                likelihood = 0.5*np.log(inv_cov_det) - 0.5*diff.T @ inv_cov @ diff
                post = prior + likelihood
                posts.append(post)
            pred = self.classes[np.argmax(posts)]
            preds.append(pred)
        return np.array(preds)