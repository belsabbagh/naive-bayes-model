import numpy as np
import pandas as pd

class NaiveBayes(object):
    def __init__(self):
        self.prob = None
        self.labels = None
        
    def conditional_prob(self, df, col, label):
        return df.groupby(col)[label].apply(lambda g: g.value_counts()/len(g)).to_dict()

    def fit(self, X: pd.DataFrame, y):
        self.labels = y.unique()
        df = pd.concat([X, y], axis=1)
        prob = {col: self.conditional_prob(df, col, y.name) for col in X.columns}
        self.prob = prob

    def predict(self, X):
        if self.prob is None or self.labels is None:
            raise Exception('Model not fitted')
        def p(X, lvalue):
            res = 1
            for col, val in X.items():
                res *= self.prob[col].get((val, lvalue), 0)
            return res
        X = X.to_dict()
        label_probs = {l: p(X, l) for l in self.labels}
        return max(label_probs, key=label_probs.get)
