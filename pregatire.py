import numpy as np

from incarcare import incarca_train, incarca_test
from features import extrage_features

folder = 'data'

#salvam datele o singura data ca sa nu le tot recalculam mereu
def pregateste_si_salveaza():
    print('Incarc train...')
    imagini_train, etichete = incarca_train(folder)
    print('Extrag features LBP train...')
    features_train = extrage_features(imagini_train)

    
    np.save('rezultate/features_train.npy', features_train)
    np.save('rezultate/etichete_train.npy', etichete)
    print('Salvat train:', features_train.shape)

    print('Incarc test...')
    imagini_test, nume_fisiere = incarca_test(folder)
    print('Extrag features LBP test...')
    features_test = extrage_features(imagini_test)

    np.save('rezultate/features_test.npy', features_test)
    np.save('rezultate/nume_test.npy', nume_fisiere)
    print('Salvat test:', features_test.shape)


if __name__ == '__main__':
    pregateste_si_salveaza()