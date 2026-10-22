import numpy as np
from sklearn.naive_bayes import MultinomialNB


def discretizeaza(date, numar_bins):

    intervale = np.linspace(start = 0, stop = 255, num = numar_bins)
    date_discrete = np.digitize(date, intervale)

    return date_discrete


def antreneaza_si_prezice(X_train, y_train, X_val, numar_bins = 5):
    X_train_d = discretizeaza(X_train, numar_bins)
    X_val_d = discretizeaza(X_val, numar_bins)

    model = MultinomialNB()
    model.fit(X_train_d, y_train)

    predictii = model.predict(X_val_d)

    return predictii