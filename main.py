import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import chi2_contingency

from src import base_test

def vectorize(data, vectorizer):
    return vectorizer.fit_transform(data)


def tfidf(X, y):
    tfidf = TfidfVectorizer()
    return vectorize(X, tfidf).toarray()


def chi2_select(X, y):
    y = y.map({'ham': 0, 'spam': 1})
    tfidf = TfidfVectorizer()
    for category_id in sorted(y.unique()):
        features_chi2 = chi2(vectorize(X, tfidf).toarray(), y == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names_out())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-2:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))

if __name__ == '__main__':
    print(f'TFIDF Accuracy: {round( base_test(tfidf), 2)}')
    print(f'Chi^2 Accuracy: {round(base_test(chi2_select), 2)}')
