from sklearn.model_selection import KFold, cross_val_score


def validate(model, X, y):
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    return cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='accuracy')
    