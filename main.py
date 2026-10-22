import numpy as np
from incarcare import incarca_train
from validare import imparte_train_validare, acuratete, matrice_confuzie, save_mc
from model_knn import ClasificatorKnn
from model_bayes import antreneaza_si_prezice
from features import extrage_features
from model_svm import antreneaza_svm, prezice_svm

folder = 'data'

def ruleaza_knn():
    print('incarc datele...')
    imagini, etichete = incarca_train(folder)

    X_train, y_train, X_val, y_val = imparte_train_validare(imagini, etichete)
    print('Train', X_train.shape, 'Validare: ', X_val.shape)

    nr_train = 2000
    nr_val = 300

    X_train_mic = X_train[:nr_train]
    y_train_mic = y_train[:nr_train]
    X_val_mic = X_val[:nr_val]
    y_val_mic = y_val[:nr_val]

    model = ClasificatorKnn(X_train_mic, y_train_mic)

    print('Clasific validarea cu kNN (k=2, L2)...')
    predictii = model.clasifica_multe(X_val_mic, numar_vecini=2, metrica='l2')

    acc = acuratete(y_val_mic, predictii)
    print('Acuratete kNN (k=2, L2) pe subset validare:', acc)

    C = matrice_confuzie(y_val_mic, predictii)
    print('Matrice de confuzie:')
    print(C)
    save_mc(C, 'confuzie_knn_k2_l2.png', titlu='kNN k=2 L2')


def ruleaza_bayes():
    print('Incarc datele...')
    imagini, etichete = incarca_train(folder)

    X_train, y_train, X_val, y_val = imparte_train_validare(imagini, etichete)
    print('Train:', X_train.shape, 'Validare:', X_val.shape)


    print('Antrenez Naive Bayes...')
    predictii = antreneaza_si_prezice(X_train, y_train, X_val, numar_bins=5)

    acc = acuratete(y_val, predictii)
    print('Acuratete Naive Bayes:', acc)

    C = matrice_confuzie(y_val, predictii)
    print('Matrice de confuzie:')
    print(C)
    save_mc(C, 'confuzie_bayes_bins5.png', titlu='Naive Bayes bins=5')


def ruleaza_svm_lbp():
    print('incarc datele...')
    imagini, etichete = incarca_train(folder)

    print('Extrag features LBP...')
    features = extrage_features(imagini)
    print('Features:', features.shape) 

    # folosim features
    X_train, y_train, X_val, y_val = imparte_train_validare(features, etichete)
    print('Train:', X_train.shape, 'Validare:', X_val.shape)

    
    print('Antrenez SVM (C=5, RBF)...')
    model, scaler = antreneaza_svm(X_train, y_train, C=5, kernel='rbf')

    # prezic pe validare
    predictii = prezice_svm(model, scaler, X_val)

    acc = acuratete(y_val, predictii)
    print('Acuratete SVM+LBP (C=5):', acc)

    C = matrice_confuzie(y_val, predictii)
    print('Matrice de confuzie:')
    print(C)
    save_mc(C, 'confuzie_svm_lbp_c5.png', titlu='SVM+LBP C=5')


if __name__ == '__main__':
    #ruleaza_knn()
    #ruleaza_bayes()
    
    # OBS: Trebuie intai rulat pregatire.py pentru a extrage feature-uri lbp
    # altfel SVM ul va da acuratete mult mai mica
    # pentru rulare CNN si generare submisie, se foloseste submisie.py
    ruleaza_svm_lbp()