import os 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def imparte_train_validare(imagini, etichete, procent_validare = 0.2, seed = 42):

    numar_total = imagini.shape[0]

    indexi = np.arange(numar_total)

    generator =  np.random.RandomState(seed)
    generator.shuffle(indexi)

    numar_validare = int(numar_total * procent_validare)

    indexi_validare = indexi[ :numar_validare]
    indexi_train = indexi[numar_validare: ]

    X_train = imagini[indexi_train]
    y_train = etichete[indexi_train]

    X_val = imagini[indexi_validare]
    y_val = etichete[indexi_validare]

    return X_train, y_train, X_val, y_val



def acuratete(y_adevarat, y_prezis):
    return np.mean(y_adevarat == y_prezis)


def matrice_confuzie(y_adevarat, y_prezis, numar_clase = 5):

    C = np.zeros((numar_clase, numar_clase), dtype=int)

    for adevarat, prezis in zip(y_adevarat, y_prezis):
        C[adevarat - 1, prezis - 1] += 1
    
    return C

def save_mc(C, nume_fisier, titlu = "Matrice de confuzie"):

    plt.figure()
    plt.imshow(C, cmap='Blues')
    plt.title(titlu)
    plt.xlabel('Clasa prezisa')
    plt.ylabel('Clasa reala')

    nr_clase = C.shape[0]

    for i in range(nr_clase):
        for j in range(nr_clase):
            plt.text(j, i, str(C[i, j]), ha='center', va='center')
        
    plt.xticks(range(nr_clase), range(1, nr_clase + 1))
    plt.yticks(range(nr_clase), range(1, nr_clase + 1))
    plt.colorbar()

    plt.savefig(os.path.join('rezultate', nume_fisier))
    plt.close()


if __name__ == '__main__':
    y_real = np.array([1, 2, 3, 1, 2, 3])
    y_pred = np.array([1, 2, 1, 1, 3, 3])

    print('Acuratete test: ', acuratete(y_real, y_pred))

    C = matrice_confuzie(y_real, y_pred, numar_clase = 3)
    print('Matrice de confuzie test: ', C, sep='\n')

