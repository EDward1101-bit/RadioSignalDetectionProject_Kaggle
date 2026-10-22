import os
import numpy as np
import pandas as pd
from PIL import Image

def incarcare_imagini(lista_nume, folder_imagini):
    # adun vectorul fiecarei imagini
    lista_vectori = [] 

    for nume_fisier in lista_nume:

        cale = os.path.join(folder_imagini, nume_fisier)

        imagine = Image.open(cale).convert('L')

        matrice = np.asarray(imagine)
        vector = matrice.reshape(-1)

        lista_vectori.append(vector)

    return np.array(lista_vectori)

def incarca_train(folder_radacina):
    cale_csv = os.path.join(folder_radacina, 'train.csv')
    tabel = pd.read_csv(cale_csv)

    nume_fisier = tabel['id'].values
    etichete = tabel['label'].values

    folder_train = os.path.join(folder_radacina, 'train')
    imagini = incarcare_imagini(nume_fisier, folder_train)

    return imagini, etichete

def incarca_test(folder_radacina):
    
    cale_csv = os.path.join(folder_radacina, 'test.csv')
    tabel = pd.read_csv(cale_csv)

    nume_fisiere = tabel['id'].values
    folder_test = os.path.join(folder_radacina, 'test')
    imagini = incarcare_imagini(nume_fisiere, folder_test)

    return imagini, nume_fisiere

if __name__ == '__main__':
    folder = 'data'

    X_train, y_train = incarca_train(folder)
    print('Imagini train:', X_train.shape)
    print('Etichete train:', y_train.shape)   
    print('Primele etichete:', y_train[:10])
    print('Pixeli min/max:', X_train.min(), X_train.max())

