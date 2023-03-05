import numpy as np
import pandas as pd

from timeit import default_timer
from sklearn.naive_bayes import GaussianNB

from src.kfold_cross_validation import validate


def base_test(preprocess):
    print('Loading data...')
    df = pd.read_csv('data/spam.csv')[['v1', 'v2']]
    X, y = df['v2'].values, df['v1']
    start = default_timer()
    X = preprocess(X, y)
    print(f'Preprocessing took {round(default_timer() - start, 2)} seconds')
    start = default_timer()
    res = validate(GaussianNB(), X, y, 100)
    print(f'Validation took {round(default_timer() - start, 2)} seconds')
    return np.sqrt(np.mean(np.absolute(res)))
