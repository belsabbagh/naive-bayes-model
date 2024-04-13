import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk

nltk.download("stopwords")

# Read the data
df = pd.read_csv("data/spam.csv", encoding="ISO 8859-15")[["category", "text"]]

# Remove stopwords
stop_words = set(stopwords.words("english"))
df["text"] = df["text"].apply(
    lambda x: " ".join(
        [word for word in word_tokenize(x) if word.lower() not in stop_words]
    )
)

X, y = (
    df["text"].values,
    df["category"].replace({"ham": "0", "spam": "1"}).astype(int).values,
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def score_test(y_pred, y_test):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
    }


# Benchmark TF-IDF method
def benchmark_tfidf():
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    return score_test(y_pred, y_test)


# Benchmark Count method
def benchmark_count():
    count = CountVectorizer()
    X_train_count = count.fit_transform(X_train)
    X_test_count = count.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_count, y_train)
    y_pred = model.predict(X_test_count)
    return score_test(y_pred, y_test)


if __name__ == "__main__":
    print(benchmark_tfidf())
    print(benchmark_count())
