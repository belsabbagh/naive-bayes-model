from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def __vectorize(data, vectorizer):
    return vectorizer.fit_transform(data)


def tfidf(X, y):
    tfidf = TfidfVectorizer(analyzer='word')
    return __vectorize(X, tfidf).toarray()


def chi2_select(X, y):
    return SelectKBest(chi2, k=64).fit_transform(
        __vectorize(X, CountVectorizer(analyzer='word')).toarray(),
        y
    )
