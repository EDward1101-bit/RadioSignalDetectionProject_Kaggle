import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def antreneaza_svm(X_train, y_train, C=5, kernel='rbf', gamma = 1.5):
    
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)

    # gamma controleaza cat de flexibila e granita SVM-ului
    # 1/(nr_features * varianta datelor)
    val_scale = 1.0 / (X_train_n.shape[1] * X_train_n.var())
    gamma = val_scale * gamma

    model = SVC(C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train_n, y_train)

    return model, scaler


def prezice_svm(model, scaler, X):
    X_n = scaler.transform(X)        
    predictii = model.predict(X_n)
    return predictii