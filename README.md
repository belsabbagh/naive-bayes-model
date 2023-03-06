# Classifying Emails as Spam or Not Spam Using Naive Bayes

## Introduction

This project is a simple implementation of a Naive Bayes classifier to classify emails as spam or not spam. The dataset used contains 5,572 emails, 3,281 of which are spam and 2,291 of which are not spam. The dataset is split into a training set and a test set, with 3,000 emails in the training set and 2,572 emails in the test set. The training set is used to train the classifier, and the test set is used to evaluate the classifier.

## Vectorizer Algorithms

### TFIDF (Term Frequency Inverse Document Frequency)

- Term Frequency: The number of times a word appears in a document, divided by the total number of words in the document. Every document has its own term frequency.
- Inverse Document Frequency: The log of the number of documents divided by the number of documents that contain the word w. Every word has its own inverse document frequency.
- TFIDF: The product of the term frequency and the inverse document frequency.

Accuracy increases as the number of features increases. However, it scored lower than chi^2.

### CHI^2 (Chi-squared Test)

- Chi-squared test: A statistical hypothesis test that is used to determine whether there is a significant difference between the expected frequencies and the observed frequencies in one or more categories of a contingency table.
- The chi^2 algorithm selects the k best features based on the chi-squared test.

Accuracy was highest when the number of features was 3000.
