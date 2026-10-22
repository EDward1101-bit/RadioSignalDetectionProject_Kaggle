import numpy as np

from validare import imparte_train_validare, acuratete
from model_svm import antreneaza_svm, prezice_svm
from model_bayes import antreneaza_si_prezice
from incarcare import incarca_train
from model_knn import ClasificatorKnn
# folosim datele salvate si testam mai multi parametrii pentru modelele noastre

def load_features():
    features = np.load('rezultate/features_train.npy')
    etichete = np.load('rezultate/etichete_train.npy')
    return features, etichete


def tuning_svm():
    features, etichete = load_features()
    X_train, y_train, X_val, y_val = imparte_train_validare(features, etichete)

    print('Tuning SVM (kernel RBF)')
    print('C\t| Acuratete validare')
    print()

    rezultate = []
    for C in [1, 5, 10, 50, 100]:
        model, scaler = antreneaza_svm(X_train, y_train, C=C, kernel='rbf')
        predictii = prezice_svm(model, scaler, X_val)
        acc = acuratete(y_val, predictii)
        rezultate.append((C, acc))
        print(f'{C}\t| {acc:.6f}')

    return rezultate

# Bayes cu features LBP
def tuning_bayes():
    features, etichete = load_features()
    X_train, y_train, X_val, y_val = imparte_train_validare(features, etichete)

    print('Tuning Naive Bayes')
    print('num_bins | Acuratete validare')
    print()

    rezultate = []
    for bins in [3, 5, 7, 9, 11]:
        predictii = antreneaza_si_prezice(X_train, y_train, X_val, numar_bins=bins)
        acc = acuratete(y_val, predictii)
        rezultate.append((bins, acc))
        print(f'{bins}\t | {acc:.6f}')

    return rezultate

# Bayes cu pixeli bruti
def tuning_bayes_pixeli():
    
    print('Incarc imaginile...')
    imagini, etichete = incarca_train('data')

    X_train, y_train, X_val, y_val = imparte_train_validare(imagini, etichete)

    print('Tuning Naive Bayes pe pixeli bruti')
    print('num_bins | Acuratete validare')
    print()

    rezultate = []
    for bins in [3, 5, 7, 9, 11]:
        predictii = antreneaza_si_prezice(X_train, y_train, X_val, numar_bins=bins)
        acc = acuratete(y_val, predictii)
        rezultate.append((bins, acc))
        print(f'{bins}\t | {acc:.6f}')

    return rezultate



def tuning_knn():
    features, etichete = load_features()
    X_train, y_train, X_val, y_val = imparte_train_validare(features, etichete)

    nr_train = 3000
    nr_val = 500
    X_train_mic = X_train[:nr_train]
    y_train_mic = y_train[:nr_train]
    X_val_mic = X_val[:nr_val]
    y_val_mic = y_val[:nr_val]

    model = ClasificatorKnn(X_train_mic, y_train_mic)

    print('Tuning kNN pe features LBP')
    print('k | metrica | Acuratete validare')
    print()

    rezultate = []
    for metrica in ['l1', 'l2']:
        for k in [1, 3, 5, 7, 9]:
            predictii = model.clasifica_multe(X_val_mic, numar_vecini=k, metrica=metrica)
            acc = acuratete(y_val_mic, predictii)
            rezultate.append((k, metrica, acc))
            print(f'{k} | {metrica} | {acc:.6f}')

    return rezultate


if __name__ == '__main__':
    #tuning_bayes()
    #tuning_bayes_pixeli()
    #tuning_knn()
    tuning_svm()
