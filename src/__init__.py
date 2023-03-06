import numpy as np
import pandas as pd

from timeit import default_timer
from sklearn.naive_bayes import GaussianNB

from src.kfold_cross_validation import validate

def describe(df):
    print(df.describe())


def base_test(X, y):
    describe(pd.DataFrame(X))
    start = default_timer()
    res = validate(GaussianNB(), X, y, 100)
    print(f'Validation took {round(default_timer() - start, 2)} seconds')
    return np.sqrt(np.mean(np.absolute(res)))
