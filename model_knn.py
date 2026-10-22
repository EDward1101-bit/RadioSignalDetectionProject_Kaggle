import numpy as np

class ClasificatorKnn:

    def __init__(self, imagini_train, etichete_train):
        self.imagini_train = imagini_train
        self.etichete_train = etichete_train

    def clasifica_imagine(self, imagine_test, numar_vecini = 3, metrica = 'l2'):

        if metrica == 'l2':
            distante = np.sqrt(np.sum((self.imagini_train - imagine_test) ** 2, axis = 1))
        else:
            distante = np.sum(np.abs(self.imagini_train - imagine_test), axis = 1)
            
        indexi_sortati = np.argsort(distante)
        indexi_vecini = indexi_sortati[:numar_vecini]

        etichete_vecini = self.etichete_train[indexi_vecini]
        
        numarari = np.bincount(etichete_vecini)
        clasa_prezisa = np.argmax(numarari)

        return clasa_prezisa



    def clasifica_multe(self, imagini_test, numar_vecini = 2, metrica = 'l2'):

        predictii = []
        for i in range(imagini_test.shape[0]):
            clasa = self.clasifica_imagine(imagini_test[i], numar_vecini, metrica)
            predictii.append(clasa)
        
        return np.array(predictii)


