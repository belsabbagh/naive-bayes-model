import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

K = 3000

def tfidf(X, y):
    tfidf = TfidfVectorizer(analyzer='word')
    return pd.DataFrame(tfidf.fit_transform(X).toarray(), columns=tfidf.get_feature_names_out())


def chi2_select(X, y):
    count = CountVectorizer(analyzer='word')
    return pd.DataFrame(
        SelectKBest(chi2, k=K).fit_transform(
            count.fit_transform(X).toarray(),
            y
        ),
        columns=count.get_feature_names_out()[0:K]
    )
