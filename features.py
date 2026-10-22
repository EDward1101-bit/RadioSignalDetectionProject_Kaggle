# Pentru fiecare pixel, comparam intensitatea lui cu cei 8 vecini
# Fiecare comparatie da un bit:1 daca vecinul >= centru, altfel 0
# Cei 8 biti formeaza un cod (0-255) care descrie textura locala

import numpy as np

inaltime = 128
latime = 55

# returnam o imaginea de coduri LBP
def lbp_imagine(img_2d, raza=1):

    img = img_2d.astype(np.float64)
    H, W = img.shape
    R = raza

    
    centru = img[R : H - R, R : W - R]

    #fiecare vecin la distanta R 
    # (dy = rand, dx = coloana)
    directii = [
        (-R, -R),  # sus-stanga
        (-R,  0),  # sus
        (-R,  R),  # sus-dreapta
        ( 0,  R),  # dreapta
        ( R,  R),  # jos-dreapta
        ( R,  0),  # jos
        ( R, -R),  # jos-stanga
        ( 0, -R),  # stanga
    ]

    coduri = np.zeros_like(centru)

    for i, (dy, dx) in enumerate(directii):
        vecin = img[R + dy : H - R + dy, R + dx : W - R + dx]   # vecinul decalat cu (dy, dx)
        bit = (vecin >= centru).astype(np.float64)
        
        coduri = coduri + bit * (2 ** i)

    return coduri


def histograma_lbp(img_2d, raza=1, numar_cutii=256):
    
    coduri = lbp_imagine(img_2d, raza=raza)

    histograma, _ = np.histogram(coduri, bins=numar_cutii, range=(0, 256))
    histograma = histograma.astype(np.float64)
    histograma = histograma / (histograma.sum() + 1e-6)

    return histograma



def features_multi_scala(img_2d, scari=(1, 2, 3, 4, 5)):
    # calculez histograma LBP la mai multe raze si le lipesc cap la cap
    histograme = []
    for raza in scari:
        h = histograma_lbp(img_2d, raza=raza)
        histograme.append(h)
    return np.concatenate(histograme) 


def extrage_features(imagini, scari=(1, 2, 3, 4, 5)):
    
    numar_imagini = imagini.shape[0]
    lista_features = []

    for i in range(numar_imagini):
        img_2d = imagini[i].reshape(inaltime, latime)
        f = features_multi_scala(img_2d, scari=scari)
        lista_features.append(f)

    return np.array(lista_features)



if __name__ == '__main__':
    test = np.random.randint(0, 256, size=(20, 20)).astype(np.float64)

    coduri = lbp_imagine(test, raza=2)
    print('Coduri LBP raza=2, forma:', coduri.shape)

    f = features_multi_scala(test)
    print('forma:', f.shape, 'suma:', round(f.sum(), 5))