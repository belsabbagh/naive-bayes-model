
import pandas as pd
from src import base_test
from src.vectorizers import chi2_select, tfidf


if __name__ == '__main__':
    df = pd.read_csv('data/spam.csv')[['category', 'email']]
    X, y = df['email'].values, df['category']
    print(f'TFIDF Accuracy: {round(base_test(tfidf(X, y), y), 5)}')
    print(f'Chi^2 Accuracy: {round(base_test(chi2_select(X, y), y), 5)}')
