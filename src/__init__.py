import numpy as np
import pandas as pd

from timeit import default_timer
from sklearn.naive_bayes import MultinomialNB

from src.kfold_cross_validation import validate

def base_test(X, y):
    print(pd.DataFrame(X).describe())
    start = default_timer()
    res = validate(MultinomialNB(), X, y)
    print(f'Validation took {round(default_timer() - start, 2)} seconds')
    return np.sqrt(np.mean(np.absolute(res)))
